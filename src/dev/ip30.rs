//! SGI Octane (IP30) system chipset: HEART, Xbow, Bridge and the IOC3.
//!
//! The Octane has no MC/HPC3/IOC. The CPU talks to HEART, which holds the
//! memory controller, the interrupt controller and a 52-bit 12.5 MHz
//! counter, and is widget 8 on the Xbow crossbar. Everything else is a
//! widget: the Xbow itself is 0, the Bridge to PCI (the IP30 board) is 15,
//! graphics is 12. Each widget has a 16 MB small window at
//! 0x1000_0000 + 16 MB * widget. The PROM lives in the Bridge's external
//! flash, at 0x1fc00000 in widget 15's window, and is mapped by `Physical`
//! directly.
//!
//! Register layouts are from IRIX's `sys/RACER/heart.h`, `sys/xtalk/xbow.h`,
//! `sys/xtalk/xwidget.h`, `sys/PCI/bridge.h` and `sys/PCI/ioc3.h`.
//!
//! What is here is what the PROM needs to reach its serial prompt.
//! Registers without a model read back what was written to them. Set
//! IRIS_IP30_TRACE=1 to log the first access to each one; that is how the
//! next thing to model gets found.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::dev::z85c30::SerialBackend;
use crate::hptimer::{TimerManager, TimerReturn};
use crate::traits::{BusDevice, BusRead16, BusRead32, BusRead64, BusRead8, BUS_ERR, BUS_OK};

// --- the physical map --------------------------------------------------

pub const HEART_PIU_BASE: u32 = 0x0ff0_0000;
pub const HEART_PIU_END: u32 = 0x1000_0000;
pub const WIDGET_SPACE_BASE: u32 = 0x1000_0000;
pub const WIDGET_SPACE_END: u32 = 0x2000_0000;
const WIDGET_SIZE: u32 = 0x0100_0000;

pub const XBOW_WIDGET: u32 = 0x0;
pub const HEART_WIDGET: u32 = 0x8;
pub const BRIDGE_WIDGET: u32 = 0xf;

/// HEART's counter: the 400 MHz XIO clock / 4 / 8.
pub const HEART_COUNT_HZ: u64 = 12_500_000;

// --- widget configuration registers (xwidget.h), common to every widget --

const WIDGET_ID: u32 = 0x04;
const WIDGET_STATUS: u32 = 0x0c;
const WIDGET_CONTROL: u32 = 0x24;

/// A widget ID register: JTAG IDCODE layout, bit 0 set.
const fn widget_id(rev: u32, part: u32, mfgr: u32) -> u32 {
    (rev << 28) | (part << 12) | (mfgr << 1) | 1
}

// --- HEART -------------------------------------------------------------

// PIU register offsets (from 0x0ff00000).
const H_MODE: u32 = 0x00000;
const H_MEMCFG0: u32 = 0x00020; // ..0x38: two 32-bit bank configs each
const H_STATUS: u32 = 0x00080;
const H_IMR0: u32 = 0x10000; // ..0x10018, one per CPU
const H_SET_ISR: u32 = 0x10020;
const H_CLR_ISR: u32 = 0x10028;
const H_ISR: u32 = 0x10030;
const H_IMSR: u32 = 0x10038;
const H_CAUSE: u32 = 0x10040;
const H_COUNT: u32 = 0x20000;
const H_COMPARE: u32 = 0x30000;
const H_PRID: u32 = 0x50000;

const HEART_STAT_BIG_ENDIAN: u64 = 1 << 8;
const HEART_STAT_PROC_ACTIVE0: u64 = 1 << 4;
/// Vector 50: count == compare.
const HEART_INT_TIMER: u64 = 1 << 50;

/// HEART's five interrupt levels and the Cause.IP bit each one drives.
/// IRIX's heart.h numbers the lines from 1 ("IP3" is Cause bit 10), and
/// IP7 is the CPU's own count/compare.
const HEART_LEVELS: [(u64, u64); 5] = [
    (0x0000_0000_0000_ffff, 1 << 10), // level 0: IP2
    (0x0000_0000_ffff_0000, 1 << 11), // level 1: IP3
    (0x0003_ffff_0000_0000, 1 << 12), // level 2: IP4
    (0x0004_0000_0000_0000, 1 << 13), // level 3: IP5, the timer
    (0xfff8_0000_0000_0000, 1 << 14), // level 4: IP6, errors
];
const HEART_IP_MASK: u64 = 0x1f << 10;

/// HEART_MEMCFG: valid, size-1 and base, in 32 MB units from RAM's start.
const MEMCFG_VALID: u32 = 0x8000_0000;
const MEMCFG_SIZE_SHIFT: u32 = 16;
const MEMCFG_SIZE_MASK: u32 = 0x003f_0000;
const MEMCFG_ADDR_MASK: u32 = 0x0000_01ff;
const MEMCFG_UNIT_SHIFT: u32 = 25;
pub const IP30_RAM_BASE: u32 = 0x2000_0000;

/// Where a bank of `mb` megabytes is in HEART's MEMCFG encoding, at base
/// `base` from IP30_RAM_BASE. `None` for a bank that is not there.
fn encode_memcfg(base: u32, mb: u32) -> u32 {
    if mb == 0 {
        return 0;
    }
    let units = (mb << 20) >> MEMCFG_UNIT_SHIFT;
    MEMCFG_VALID
        | (((units.max(1) - 1) << MEMCFG_SIZE_SHIFT) & MEMCFG_SIZE_MASK)
        | ((base >> MEMCFG_UNIT_SHIFT) & MEMCFG_ADDR_MASK)
}

/// The PROM sizes a bank from where it aliases in its largest
/// configuration (MEMCFG 0x01bf/0x019f): at +256 bytes, +128 MB, and then
/// +1 GB or +64 MB, each a DIMM address line that real parts may lack. With
/// a HEART older than revision 6 (the Octane's is D, 4) the only sizes a
/// linear mirror reports correctly are 1 GB and 2 GB, so the emulated banks
/// are 1 GB; smaller ones would need each DIMM geometry's aliasing modelled.
///
/// The banks a set of MEMCFG words describes, for `Physical::remap_banks`:
/// (physical base, address mask, bytes mapped) for each of the four.
pub fn decode_memcfg(words: &[u32; 4], sizes_mb: &[u32; 4]) -> [Option<(u32, u32, u32)>; 4] {
    let mut out = [None; 4];
    for i in 0..4 {
        let w = words[i];
        if w & MEMCFG_VALID == 0 || sizes_mb[i] == 0 {
            continue;
        }
        let size = ((((w & MEMCFG_SIZE_MASK) >> MEMCFG_SIZE_SHIFT) + 1) as u64) << MEMCFG_UNIT_SHIFT;
        let size = size.min(1 << 31) as u32;
        let base = IP30_RAM_BASE as u64 + (((w & MEMCFG_ADDR_MASK) as u64) << MEMCFG_UNIT_SHIFT);
        if base >= 1 << 32 {
            continue; // beyond the 32-bit bus (see NOTES: wider physical addresses)
        }
        let base = base as u32;
        // A DIMM smaller than its MEMCFG size repeats through the rest of
        // it: the PROM sizes memory by looking for where that starts. The
        // window stops at 4 GB, where the bus does.
        let real = sizes_mb[i] << 20;
        let span = (size as u64).min((1u64 << 32) - base as u64) as u32;
        out[i] = Some((base, real.min(size) - 1, span));
    }
    out
}

struct HeartState {
    mode: u64,
    memcfg: [u32; 8],
    imr: [u64; 4],
    isr: u64,
    compare: u64,
    cause: u64,
    other: HashMap<u32, u64>,
    widget: HashMap<u32, u64>,
}

pub struct Heart {
    s: Mutex<HeartState>,
    start: Instant,
    /// The CPU's interrupt word (`MipsCore::hot.interrupts`).
    cpu_irq: AtomicPtr<AtomicU64>,
    sizes_mb: [u32; 4],
    memcfg_cb: Mutex<Option<Box<dyn Fn([Option<(u32, u32, u32)>; 4]) + Send + Sync>>>,
}

impl Heart {
    fn new(banks_mb: [u32; 4]) -> Self {
        // Banks back to back from RAM's start, as the PROM would leave them.
        let mut memcfg = [0u32; 8];
        let mut base = 0u32;
        for (i, &mb) in banks_mb.iter().enumerate() {
            memcfg[i] = encode_memcfg(base, mb);
            base += mb << 20;
        }
        Heart {
            s: Mutex::new(HeartState {
                mode: 0,
                memcfg,
                imr: [0; 4],
                isr: 0,
                compare: u64::MAX,
                cause: 0,
                other: HashMap::new(),
                widget: HashMap::new(),
            }),
            start: Instant::now(),
            cpu_irq: AtomicPtr::new(std::ptr::null_mut()),
            sizes_mb: banks_mb,
            memcfg_cb: Mutex::new(None),
        }
    }

    fn count(&self) -> u64 {
        let ns = self.start.elapsed().as_nanos() as u64;
        (ns / (1_000_000_000 / HEART_COUNT_HZ)) & ((1 << 52) - 1)
    }

    /// The banks as MEMCFG now has them.
    pub fn bank_map(&self) -> [Option<(u32, u32, u32)>; 4] {
        let s = self.s.lock();
        let w = [s.memcfg[0], s.memcfg[1], s.memcfg[2], s.memcfg[3]];
        decode_memcfg(&w, &self.sizes_mb)
    }

    /// Recompute the CPU's IP2..IP6 from ISR and CPU 0's mask.
    fn update_irq(&self, s: &HeartState) {
        let p = self.cpu_irq.load(Ordering::Acquire);
        if p.is_null() {
            return;
        }
        let pending = s.isr & s.imr[0];
        let mut ip = 0;
        for (mask, bit) in HEART_LEVELS {
            if pending & mask != 0 {
                ip |= bit;
            }
        }
        let word = unsafe { &*p };
        word.fetch_and(!(HEART_IP_MASK & !ip), Ordering::AcqRel);
        if ip != 0 {
            word.fetch_or(ip, Ordering::AcqRel);
        }
    }

    /// Count has passed compare: raise the timer vector.
    fn poll_timer(&self) {
        let now = self.count();
        let mut s = self.s.lock();
        if now >= s.compare && s.isr & HEART_INT_TIMER == 0 {
            s.isr |= HEART_INT_TIMER;
            self.update_irq(&s);
        }
    }

    fn piu_read(&self, off: u32) -> u64 {
        let s = self.s.lock();
        match off {
            H_MODE => s.mode,
            o if (H_MEMCFG0..H_MEMCFG0 + 0x20).contains(&o) => {
                let i = ((o - H_MEMCFG0) / 8) as usize * 2;
                ((s.memcfg[i] as u64) << 32) | s.memcfg[i + 1] as u64
            }
            H_STATUS => HEART_STAT_BIG_ENDIAN | HEART_STAT_PROC_ACTIVE0 | HEART_WIDGET as u64,
            o if (H_IMR0..H_IMR0 + 0x20).contains(&o) => s.imr[((o - H_IMR0) / 8) as usize],
            H_ISR => s.isr,
            H_IMSR => s.isr & s.imr[0],
            H_CAUSE => s.cause,
            H_COUNT => {
                drop(s);
                self.count()
            }
            H_COMPARE => s.compare,
            H_PRID => 0,
            o => *s.other.get(&o).unwrap_or(&0),
        }
    }

    fn piu_write(&self, off: u32, val: u64) {
        let mut s = self.s.lock();
        match off {
            H_MODE => s.mode = val,
            o if (H_MEMCFG0..H_MEMCFG0 + 0x20).contains(&o) => {
                let i = ((o - H_MEMCFG0) / 8) as usize * 2;
                s.memcfg[i] = (val >> 32) as u32;
                s.memcfg[i + 1] = val as u32;
                let w = [s.memcfg[0], s.memcfg[1], s.memcfg[2], s.memcfg[3]];
                drop(s);
                let map = decode_memcfg(&w, &self.sizes_mb);
                if let Some(cb) = self.memcfg_cb.lock().as_ref() {
                    cb(map);
                }
                return;
            }
            o if (H_IMR0..H_IMR0 + 0x20).contains(&o) => s.imr[((o - H_IMR0) / 8) as usize] = val,
            H_SET_ISR => s.isr |= val,
            H_CLR_ISR => s.isr &= !val,
            H_CAUSE => s.cause &= !val,
            H_COMPARE => {
                s.compare = val & ((1 << 52) - 1);
                s.isr &= !HEART_INT_TIMER;
            }
            H_COUNT => {}
            o => {
                s.other.insert(o, val);
            }
        }
        self.update_irq(&s);
    }

    fn widget_read(&self, off: u32) -> u64 {
        match off & !7 {
            0x00 => widget_id(4, 0xc001, 0x036) as u64, // HEART rev D
            o => *self.s.lock().widget.get(&o).unwrap_or(&0),
        }
    }

    fn widget_write(&self, off: u32, val: u64) {
        self.s.lock().widget.insert(off & !7, val);
    }
}

// --- Xbow --------------------------------------------------------------

const XB_LINK_BASE: u32 = 0x100;
const XB_LINK_SIZE: u32 = 0x40;
const XB_LINK_STATUS: u32 = 0x14;
const XB_LINK_AUX_STATUS: u32 = 0x3c;
const XB_STAT_LINKALIVE: u32 = 0x8000_0000;
const XB_AUX_STAT_PRESENT: u32 = 0x20;

pub struct Xbow {
    present: [bool; 16],
    regs: Mutex<HashMap<u32, u32>>,
}

impl Xbow {
    fn new(present: &[u32]) -> Self {
        let mut p = [false; 16];
        for &w in present {
            p[w as usize] = true;
        }
        Xbow { present: p, regs: Mutex::new(HashMap::new()) }
    }

    /// 32-bit registers, each in the low word of a doubleword.
    fn read32(&self, off: u32) -> u32 {
        if (XB_LINK_BASE..XB_LINK_BASE + 8 * XB_LINK_SIZE).contains(&off) {
            let port = 8 + (off - XB_LINK_BASE) / XB_LINK_SIZE;
            let reg = (off - XB_LINK_BASE) % XB_LINK_SIZE;
            let up = self.present[port as usize];
            match reg {
                XB_LINK_STATUS => return if up { XB_STAT_LINKALIVE } else { 0 },
                XB_LINK_AUX_STATUS => return if up { XB_AUX_STAT_PRESENT } else { 0 },
                _ => {}
            }
        }
        match off {
            WIDGET_ID => widget_id(3, 0x0, 0x0), // crossbow rev 1.2
            o => *self.regs.lock().get(&o).unwrap_or(&0),
        }
    }

    fn write32(&self, off: u32, val: u32) {
        self.regs.lock().insert(off, val);
    }
}

// --- PCI devices behind the Bridge -------------------------------------

/// One PCI function's configuration space, with BAR sizing.
struct PciConfig {
    regs: [u32; 64],
    /// Size of each BAR (power of two), 0 when absent.
    bar_size: [u32; 6],
}

impl PciConfig {
    fn new(vendor: u16, device: u16, class: u32, bars: [u32; 6]) -> Self {
        let mut regs = [0u32; 64];
        regs[0] = (device as u32) << 16 | vendor as u32;
        regs[2] = class << 8;
        PciConfig { regs, bar_size: bars }
    }

    fn read(&self, reg: usize) -> u32 {
        self.regs[reg & 63]
    }

    fn write(&mut self, reg: usize, val: u32) {
        let reg = reg & 63;
        match reg {
            0 | 2 => {} // IDs and class are read-only
            4..=9 => {
                let sz = self.bar_size[reg - 4];
                self.regs[reg] = if sz == 0 { 0 } else { val & !(sz - 1) };
            }
            _ => self.regs[reg] = val,
        }
    }
}

// --- IOC3: just the two UARTs for now ----------------------------------

const IOC3_MCR: u32 = 0x30;
const IOC3_UARTA: u32 = 0x20178;
const IOC3_UARTB: u32 = 0x20170;

/// A 16550 as the PROM drives it: polled, no FIFOs to speak of.
struct Uart {
    backend: Arc<dyn SerialBackend>,
    ier: u8,
    lcr: u8,
    mcr: u8,
    scr: u8,
    dll: u8,
    dlm: u8,
    rx: Option<u8>,
}

impl Uart {
    fn new(backend: Arc<dyn SerialBackend>) -> Self {
        Uart { backend, ier: 0, lcr: 0, mcr: 0, scr: 0, dll: 0, dlm: 0, rx: None }
    }

    fn poll_rx(&mut self) {
        if self.rx.is_none() {
            if let Ok(b) = self.backend.recv_byte() {
                self.rx = Some(b);
            }
        }
    }

    fn read(&mut self, reg: u32) -> u8 {
        let dlab = self.lcr & 0x80 != 0;
        match reg {
            0 if dlab => self.dll,
            0 => {
                self.poll_rx();
                self.rx.take().unwrap_or(0)
            }
            1 if dlab => self.dlm,
            1 => self.ier,
            2 => 0xc1, // no interrupt pending, FIFOs on
            3 => self.lcr,
            4 => self.mcr,
            5 => {
                self.poll_rx();
                0x60 | self.rx.is_some() as u8 // THRE | TEMT | DR
            }
            6 => 0xb0, // DCD, DSR, CTS
            _ => self.scr,
        }
    }

    fn write(&mut self, reg: u32, val: u8) {
        let dlab = self.lcr & 0x80 != 0;
        match reg {
            0 if dlab => self.dll = val,
            0 => self.backend.send_byte(val),
            1 if dlab => self.dlm = val,
            1 => self.ier = val,
            2 => {} // FCR
            3 => self.lcr = val,
            4 => self.mcr = val,
            7 => self.scr = val,
            _ => {}
        }
    }
}

/// The DS1687 on the IOC3's byte bus: its address port is byte bus 1
/// (0xa0000), its data port byte bus 2 (0xc0000), as OpenBSD's dsrtc.c
/// has it for IP30 (the SuperIO's own RTC port is unused here). A clock with 114 bytes of RAM in bank 0 and 128 more
/// through the extended registers of bank 1 (register A's DV0). IRIX keeps
/// its PROM environment here (sys/RACER/IP30nvram.h). The clock reads the
/// host's time; the RAM is saved to `nvram` on every change.
const IOC3_RTC_AD: u32 = 0xa0000;
const IOC3_RTC_DATA: u32 = 0xc0000;

pub struct Ds1687 {
    reg: [u8; 128],
    ext: [u8; 128],
    index: u8,
    ext_addr: u8,
    path: Option<std::path::PathBuf>,
}

impl Ds1687 {
    pub fn new(path: Option<std::path::PathBuf>) -> Self {
        let mut d = Ds1687 { reg: [0; 128], ext: [0; 128], index: 0, ext_addr: 0, path };
        if let Some(saved) = d.path.as_ref().and_then(|p| std::fs::read(p).ok()) {
            if saved.len() == 256 {
                d.reg.copy_from_slice(&saved[..128]);
                d.ext.copy_from_slice(&saved[128..]);
            }
        }
        d.reg[0x0b] |= 0x02; // 24-hour
        d
    }

    fn save(&self) {
        if let Some(p) = &self.path {
            let mut all = self.reg.to_vec();
            all.extend_from_slice(&self.ext);
            if let Err(e) = std::fs::write(p, all) {
                eprintln!("iris: nvram: could not save {}: {e}", p.display());
            }
        }
    }

    fn bank1(&self) -> bool {
        self.reg[0x0a] & 0x10 != 0
    }

    /// Host time in the chip's format: BCD unless register B's DM.
    fn clock(&self, r: u8) -> u8 {
        let t = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs()) as i64;
        let days = t.div_euclid(86400);
        let secs = t.rem_euclid(86400);
        // civil from days (Howard Hinnant's algorithm)
        let z = days + 719468;
        let era = z.div_euclid(146097);
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let day = doy - (153 * mp + 2) / 5 + 1;
        let month = if mp < 10 { mp + 3 } else { mp - 9 };
        let year = yoe + era * 400 + (month <= 2) as i64;
        let v = match r {
            0x00 => secs % 60,
            0x02 => secs / 60 % 60,
            0x04 => secs / 3600,
            0x06 => (days + 4).rem_euclid(7) + 1,
            0x07 => day,
            0x08 => month,
            0x09 => year % 100,
            _ => 0,
        } as u8;
        if self.reg[0x0b] & 0x04 != 0 { v } else { (v / 10) << 4 | v % 10 }
    }

    fn read(&mut self) -> u8 {
        let i = self.index & 0x7f;
        match i {
            0x00 | 0x02 | 0x04 | 0x06..=0x09 => self.clock(i),
            0x0a => self.reg[0x0a] & 0x7f, // never mid-update
            0x0c => 0,
            0x0d => 0x80, // VRT: the battery is fine
            0x40..=0x7f if self.bank1() => match i {
                0x40 => 0x71, // model: DS1687
                0x48 => if self.reg[0x0b] & 0x04 != 0 { 20 } else { 0x20 }, // century
                0x50 => self.ext_addr,
                0x53 => self.ext[(self.ext_addr & 0x7f) as usize],
                _ => 0,
            },
            _ => self.reg[i as usize],
        }
    }

    fn write(&mut self, val: u8) {
        let i = self.index & 0x7f;
        match i {
            0x00..=0x09 | 0x0c | 0x0d => {} // the clock follows the host
            0x40..=0x7f if self.bank1() => match i {
                0x50 => self.ext_addr = val,
                0x53 => {
                    self.ext[(self.ext_addr & 0x7f) as usize] = val;
                    self.save();
                }
                _ => {}
            },
            _ => {
                self.reg[i as usize] = val;
                if i >= 0x0e {
                    self.save();
                }
            }
        }
    }
}

struct Ioc3 {
    uart_a: Uart,
    uart_b: Uart,
    rtc: Ds1687,
    mcr: OneWire,
    regs: HashMap<u32, u32>,
}

impl Ioc3 {
    fn uart(&mut self, off: u32) -> Option<(&mut Uart, u32)> {
        if (IOC3_UARTA..IOC3_UARTA + 8).contains(&off) {
            Some((&mut self.uart_a, off - IOC3_UARTA))
        } else if (IOC3_UARTB..IOC3_UARTB + 8).contains(&off) {
            Some((&mut self.uart_b, off - IOC3_UARTB))
        } else {
            None
        }
    }

    fn read8(&mut self, off: u32) -> u8 {
        if let Some((u, r)) = self.uart(off) {
            return u.read(r);
        }
        match off & !0x1ffff {
            IOC3_RTC_AD => return self.rtc.index,
            IOC3_RTC_DATA => return self.rtc.read(),
            _ => {}
        }
        let w = *self.regs.get(&(off & !3)).unwrap_or(&0);
        (w >> (8 * (3 - (off & 3)))) as u8
    }

    fn write8(&mut self, off: u32, val: u8) {
        if let Some((u, r)) = self.uart(off) {
            return u.write(r, val);
        }
        match off & !0x1ffff {
            IOC3_RTC_AD => return self.rtc.index = val,
            IOC3_RTC_DATA => return self.rtc.write(val),
            _ => {}
        }
        let sh = 8 * (3 - (off & 3));
        let w = self.regs.entry(off & !3).or_insert(0);
        *w = (*w & !(0xff << sh)) | (val as u32) << sh;
    }

    fn read32(&mut self, off: u32) -> u32 {
        match off & !3 {
            IOC3_MCR => self.mcr.read(),
            o => *self.regs.get(&o).unwrap_or(&0),
        }
    }

    fn write32(&mut self, off: u32, val: u32) {
        match off & !3 {
            IOC3_MCR => self.mcr.write(val),
            o => {
                self.regs.insert(o, val);
            }
        }
    }
}

// --- the NIC ("number in a can") 1-wire masters ------------------------

/// SGI's 1-wire master, as on the Bridge (BRIDGE_NIC), HEART and IOC3
/// (IOC3_MCR): software writes a slot's low time and sample time and reads
/// back DONE and the line as sampled (Linux's drivers/w1/masters/sgi_w1.c:
/// MCR_PACK(pulse, sample) = pulse << 10 | sample << 2). A low time of
/// 480 µs or more is a reset, under 15 µs a 1 (or a read slot: the same
/// thing on the wire), anything between a 0.
const MCR_RD_DATA: u32 = 1 << 0;
const MCR_DONE: u32 = 1 << 1;

#[derive(Default)]
struct OneWire {
    last: u32,
    line: bool,
    dev: Option<Ds2502>,
}

impl OneWire {
    fn with(dev: Ds2502) -> Self {
        OneWire { dev: Some(dev), ..Default::default() }
    }

    fn write(&mut self, val: u32) {
        self.last = val;
        let pulse = (val >> 10) & 0x3ff;
        self.line = match &mut self.dev {
            _ if pulse >= 480 => !self.dev.as_mut().is_some_and(|d| d.reset()),
            Some(d) => d.slot(pulse < 15),
            None => true,
        };
    }

    fn read(&self) -> u32 {
        (self.last & !(MCR_RD_DATA | MCR_DONE)) | MCR_DONE | self.line as u32
    }
}

/// Dallas's 1-wire CRC-8 (x^8 + x^5 + x^4 + 1).
fn crc8(bytes: &[u8]) -> u8 {
    let mut crc = 0u8;
    for &b in bytes {
        let mut b = b;
        for _ in 0..8 {
            let mix = (crc ^ b) & 1;
            crc >>= 1;
            if mix != 0 {
                crc ^= 0x8c;
            }
            b >>= 1;
        }
    }
    crc
}

/// Dallas's CRC-16 (x^16 + x^15 + x^2 + 1), as TMEX pages carry it.
fn crc16(bytes: &[u8]) -> u16 {
    let mut crc = 0u16;
    for &b in bytes {
        crc ^= b as u16;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { crc >> 1 ^ 0xa001 } else { crc >> 1 };
        }
    }
    crc
}

/// What a 1-wire part is doing between slots.
enum WireState {
    Idle,
    /// Taking `n` bits, least significant first (`i` so far), then acting
    /// on them.
    Recv { n: u32, i: u32, got: u64, then: Next },
    /// Sending `bits` from `i` on; after them, `then`.
    Send { bits: Vec<bool>, i: usize, then: Next },
    /// Search ROM at ROM bit `bit`: phase 0 sends it, 1 its complement,
    /// 2 takes the master's choice.
    Search { bit: u32, phase: u8 },
}

#[derive(Clone, Copy, Debug)]
enum Next {
    RomCmd,
    MatchRom,
    FuncCmd,
    ReadAddr(u8),
    Idle,
}

/// A DS2502-family add-only memory on the bus: an SGI NIC. Answers the
/// ROM commands (read 0x33, match 0x55, skip 0xcc, search 0xf0) and
/// Read Memory (0xf0) and Read Status (0xaa).
pub struct Ds2502 {
    rom: [u8; 8],
    mem: Vec<u8>,
    state: WireState,
}

fn bits_of(bytes: &[u8]) -> Vec<bool> {
    bytes.iter().flat_map(|&b| (0..8).map(move |i| b >> i & 1 != 0)).collect()
}

impl Ds2502 {
    pub fn new(family: u8, serial: [u8; 6], mem: Vec<u8>) -> Self {
        let mut rom = [0u8; 8];
        rom[0] = family;
        rom[1..7].copy_from_slice(&serial);
        rom[7] = crc8(&rom[..7]);
        Ds2502 { rom, mem, state: WireState::Idle }
    }

    /// The MAC NIC on the IOC3: a DS1981U (family 0x91). The PROM, like
    /// Linux's ioc3-eth.c, reads 14 bytes after Read Memory at 0, the
    /// first of them the CRC8 of the command, into nic[13..0], and takes
    /// the station address from nic[2..8]. That puts it reversed at memory
    /// 5..11, the end of a TMEX page: a length byte (10), the data, and the
    /// inverted CRC16 of both.
    pub fn mac_nic(mac: [u8; 6]) -> Self {
        let mut mem = vec![0xffu8; 128];
        mem[0] = 10;
        mem[1..5].fill(0);
        for j in 0..6 {
            mem[10 - j] = mac[j];
        }
        let crc = !crc16(&mem[..11]);
        mem[11] = crc as u8;
        mem[12] = (crc >> 8) as u8;
        // A UniqueWare 0x91 part's serial starts 0x5e7 (OpenBSD's owmac.c
        // matches on it); the rest is from the MAC.
        Ds2502::new(0x91, [mac[5], mac[4], mac[3], mac[2], 0x70 | (mac[1] & 0x0f), 0x5e], mem)
    }

    fn rom_bit(&self, bit: u32) -> bool {
        self.rom[(bit / 8) as usize] >> (bit % 8) & 1 != 0
    }

    /// A reset: present, and waiting for a ROM command.
    fn reset(&mut self) -> bool {
        if std::env::var_os("IRIS_IP30_TRACE").is_some() {
            let sent = match &self.state { WireState::Send { i, .. } => *i as i64, _ => -1 };
            eprintln!("ip30: 1-wire {:02x}: reset (after {} bits sent)", self.rom[0], sent);
        }
        self.state = WireState::Recv { n: 8, i: 0, got: 0, then: Next::RomCmd };
        true
    }

    fn recv(n: u32, then: Next) -> WireState {
        WireState::Recv { n, i: 0, got: 0, then }
    }

    /// What follows `n` bits arriving for `then`.
    fn act(&mut self, then: Next, got: u64) -> WireState {
        if std::env::var_os("IRIS_IP30_TRACE").is_some() {
            eprintln!("ip30: 1-wire {:02x}: {:?} {:#x}", self.rom[0], then, got);
        }
        match then {
            Next::RomCmd => match got as u8 {
                0x33 => WireState::Send { bits: bits_of(&self.rom), i: 0, then: Next::FuncCmd },
                0x55 => Self::recv(64, Next::MatchRom),
                0xcc => Self::recv(8, Next::FuncCmd),
                0xf0 => WireState::Search { bit: 0, phase: 0 },
                _ => WireState::Idle,
            },
            Next::MatchRom if got == u64::from_le_bytes(self.rom) => Self::recv(8, Next::FuncCmd),
            Next::MatchRom => WireState::Idle,
            Next::FuncCmd => match got as u8 {
                c @ (0xf0 | 0xaa) => Self::recv(16, Next::ReadAddr(c)),
                _ => WireState::Idle,
            },
            Next::ReadAddr(cmd) => {
                let addr = got as u16;
                let mut out = vec![crc8(&[cmd, addr as u8, (addr >> 8) as u8])];
                if cmd == 0xf0 {
                    out.extend(self.mem.iter().skip(addr as usize));
                } else {
                    out.extend(std::iter::repeat(0xff).take(8usize.saturating_sub(addr as usize)));
                }
                WireState::Send { bits: bits_of(&out), i: 0, then: Next::Idle }
            }
            Next::Idle => WireState::Idle,
        }
    }

    /// One slot: `master` is what the master drove (1 also for a read
    /// slot); returns what the line reads.
    fn slot(&mut self, master: bool) -> bool {
        let state = std::mem::replace(&mut self.state, WireState::Idle);
        let (line, next) = match state {
            WireState::Idle => (master, WireState::Idle),
            WireState::Recv { n, i, got, then } => {
                let got = got | (master as u64) << i;
                if i + 1 == n {
                    (master, self.act(then, got))
                } else {
                    (master, WireState::Recv { n, i: i + 1, got, then })
                }
            }
            WireState::Send { bits, i, then } => {
                // Past the end the part has let go: the line floats high.
                let b = bits.get(i).copied().unwrap_or(true);
                if i % 64 == 63 && std::env::var_os("IRIS_IP30_TRACE").is_some() {
                    eprintln!("ip30: 1-wire {:02x}: sent {} bits", self.rom[0], i + 1);
                }
                let next = if i + 1 >= bits.len() {
                    match then {
                        Next::FuncCmd => Self::recv(8, Next::FuncCmd),
                        _ => WireState::Send { bits, i: i + 1, then },
                    }
                } else {
                    WireState::Send { bits, i: i + 1, then }
                };
                (master && b, next)
            }
            WireState::Search { bit, phase } => {
                let b = self.rom_bit(bit);
                match phase {
                    0 => (master && b, WireState::Search { bit, phase: 1 }),
                    1 => (master && !b, WireState::Search { bit, phase: 2 }),
                    _ if master != b => (master, WireState::Idle),
                    _ if bit == 63 => (master, Self::recv(8, Next::FuncCmd)),
                    _ => (master, WireState::Search { bit: bit + 1, phase: 0 }),
                }
            }
        };
        self.state = next;
        line
    }
}

// --- Bridge ------------------------------------------------------------

const BRIDGE_CONFIG_BASE: u32 = 0x20000;
const BRIDGE_CONFIG_END: u32 = 0x28000;
const BRIDGE_DEVIO0: u32 = 0x20_0000;
const BRIDGE_DEVIO_END: u32 = 0xc0_0000;
const BRIDGE_INT_STATUS: u32 = 0x104;
const BRIDGE_NIC: u32 = 0xb4;

/// The PCI slot the IOC3 is in.
const IOC3_SLOT: usize = 2;

/// Which device's DevIO window an offset is in, and where in it.
fn devio_slot(off: u32) -> Option<(usize, u32)> {
    match off {
        0x20_0000..=0x3f_ffff => Some((0, off - 0x20_0000)),
        0x40_0000..=0x5f_ffff => Some((1, off - 0x40_0000)),
        0x60_0000..=0xbf_ffff => {
            let n = (off - 0x60_0000) >> 20;
            Some((2 + n as usize, (off - 0x60_0000) & 0xf_ffff))
        }
        _ => None,
    }
}

pub struct Bridge {
    regs: Mutex<HashMap<u32, u32>>,
    nic: Mutex<OneWire>,
    pci: Mutex<[Option<PciConfig>; 8]>,
    ioc3: Mutex<Ioc3>,
}

impl Bridge {
    fn new(rtc: Ds1687, mac: [u8; 6], console: Arc<dyn SerialBackend>, aux: Arc<dyn SerialBackend>) -> Self {
        let mut pci: [Option<PciConfig>; 8] = Default::default();
        // As on snorlax: two QL1040Bs, the IOC3, and RAD audio.
        pci[0] = Some(PciConfig::new(0x1077, 0x1020, 0x010000, [0x100, 0x1000, 0, 0, 0, 0]));
        pci[1] = Some(PciConfig::new(0x1077, 0x1020, 0x010000, [0x100, 0x1000, 0, 0, 0, 0]));
        pci[IOC3_SLOT] = Some(PciConfig::new(0x10a9, 0x0003, 0xff0000, [0x10_0000, 0, 0, 0, 0, 0]));
        pci[3] = Some(PciConfig::new(0x10a9, 0x0005, 0x040100, [0x10_0000, 0, 0, 0, 0, 0]));
        Bridge {
            regs: Mutex::new(HashMap::new()),
            nic: Mutex::new(OneWire::default()),
            pci: Mutex::new(pci),
            ioc3: Mutex::new(Ioc3 {
                uart_a: Uart::new(console),
                uart_b: Uart::new(aux),
                rtc,
                mcr: OneWire::with(Ds2502::mac_nic(mac)),
                regs: HashMap::new(),
            }),
        }
    }

    fn config_slot(off: u32) -> (usize, usize, usize) {
        let o = off - BRIDGE_CONFIG_BASE;
        ((o >> 12) as usize & 7, (o >> 8) as usize & 7, (o & 0xff) as usize)
    }

    /// A configuration dword; absent devices read as all ones.
    fn config_read(&self, off: u32) -> u32 {
        let (slot, func, reg) = Self::config_slot(off);
        match &self.pci.lock()[slot] {
            Some(c) if func == 0 => c.read(reg / 4),
            _ => 0xffff_ffff,
        }
    }

    fn config_write(&self, off: u32, val: u32) {
        let (slot, func, reg) = Self::config_slot(off);
        if let Some(c) = &mut self.pci.lock()[slot] {
            if func == 0 {
                c.write(reg / 4, val);
            }
        }
    }

    fn reg_read32(&self, off: u32) -> u32 {
        match off {
            WIDGET_ID => widget_id(4, 0xc002, 0x036), // Bridge rev D
            BRIDGE_INT_STATUS => 0,
            BRIDGE_NIC => self.nic.lock().read(),
            o => *self.regs.lock().get(&o).unwrap_or(&0),
        }
    }

    fn reg_write32(&self, off: u32, val: u32) {
        if off == BRIDGE_NIC {
            return self.nic.lock().write(val);
        }
        self.regs.lock().insert(off, val);
    }
}

// --- the PROM flash ----------------------------------------------------

/// The flash at 0x1fc00000 (Bridge external flash): one 1 MB Sharp
/// LH28F800 in sixteen 64 KB segments, RPROM in 0-2, FPROM in 3-14 and
/// the PROM's data store (PDS) in 15 (sys/RACER/sflash.h). Reads return
/// the array, or ID/status after the matching command; programming ANDs
/// a halfword in, erase sets a segment to ones. Changes are written back
/// to the `flash` file, if there is one.
pub const FLASH_BASE: u32 = 0x1fc0_0000;
pub const FLASH_SIZE: u32 = 0x10_0000;
const FLASH_SEG: usize = 64 * 1024;
const SFLASH_MFG_ID: u16 = 0x00b0;
const SFLASH_DEV_ID: u16 = 0x66a8;
const SFLASH_STATUS_READY: u16 = 0x80;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FlashMode {
    Read,
    Id,
    Status,
    Program,
    Erase,
}

struct FlashState {
    data: Vec<u8>,
    mode: FlashMode,
}

pub struct Flash {
    s: Mutex<FlashState>,
    path: Option<std::path::PathBuf>,
    /// jitv2's code-generation counter for the whole part.
    gen: AtomicU64,
}

impl Flash {
    /// From `path` if it holds a flash image, else from `image` (and
    /// `path`, when given, gets a copy).
    pub fn new(image: &[u8], path: Option<std::path::PathBuf>) -> Self {
        let saved = path.as_ref().and_then(|p| std::fs::read(p).ok())
            .filter(|d| d.len() == FLASH_SIZE as usize);
        let mut data = saved.clone().unwrap_or_else(|| image.to_vec());
        data.resize(FLASH_SIZE as usize, 0xff);
        let f = Flash {
            s: Mutex::new(FlashState { data, mode: FlashMode::Read }),
            path,
            gen: AtomicU64::new(0),
        };
        if saved.is_none() {
            f.save(&f.s.lock().data);
        }
        f
    }

    fn save(&self, data: &[u8]) {
        if let Some(p) = &self.path {
            if let Err(e) = std::fs::write(p, data) {
                eprintln!("iris: flash: could not save {}: {e}", p.display());
            }
        }
    }

    fn get16(&self, off: usize) -> u16 {
        let s = self.s.lock();
        match s.mode {
            FlashMode::Read => u16::from_be_bytes([s.data[off], s.data[off + 1]]),
            FlashMode::Id => if off & 2 == 0 { SFLASH_MFG_ID } else { SFLASH_DEV_ID },
            _ => SFLASH_STATUS_READY,
        }
    }

    fn put16(&self, off: usize, val: u16) {
        let mut s = self.s.lock();
        match s.mode {
            FlashMode::Program => {
                let old = u16::from_be_bytes([s.data[off], s.data[off + 1]]);
                let new = (old & val).to_be_bytes();
                s.data[off..off + 2].copy_from_slice(&new);
                s.mode = FlashMode::Status;
                self.gen.fetch_add(1, Ordering::Release);
                self.save(&s.data);
                return;
            }
            FlashMode::Erase if val & 0xff == 0xd0 => {
                let seg = off / FLASH_SEG * FLASH_SEG;
                s.data[seg..seg + FLASH_SEG].fill(0xff);
                s.mode = FlashMode::Status;
                self.gen.fetch_add(1, Ordering::Release);
                self.save(&s.data);
                return;
            }
            _ => {}
        }
        s.mode = match val & 0xff {
            0xff => FlashMode::Read,
            0x90 => FlashMode::Id,
            0x70 | 0x71 | 0x50 => FlashMode::Status,
            0x40 | 0x10 => FlashMode::Program,
            0x20 => FlashMode::Erase,
            _ => s.mode,
        };
        self.gen.fetch_add(1, Ordering::Release);
    }

    fn off(addr: u32) -> Option<usize> {
        (FLASH_BASE..FLASH_BASE + FLASH_SIZE).contains(&addr).then(|| (addr - FLASH_BASE) as usize)
    }
}

impl BusDevice for Flash {
    fn read8(&self, addr: u32) -> BusRead8 {
        let Some(o) = Self::off(addr) else { return BusRead8::ok(0xff) };
        BusRead8::ok((self.get16(o & !1) >> (8 * (1 - (o & 1)))) as u8)
    }
    fn read16(&self, addr: u32) -> BusRead16 {
        Self::off(addr).map_or(BusRead16::ok(0xffff), |o| BusRead16::ok(self.get16(o & !1)))
    }
    fn read32(&self, addr: u32) -> BusRead32 {
        let Some(o) = Self::off(addr & !3) else { return BusRead32::ok(!0) };
        BusRead32::ok((self.get16(o) as u32) << 16 | self.get16(o + 2) as u32)
    }
    fn read64(&self, addr: u32) -> BusRead64 {
        let hi = self.read32(addr & !7).data as u64;
        BusRead64::ok(hi << 32 | self.read32((addr & !7) + 4).data as u64)
    }
    fn write16(&self, addr: u32, val: u16) -> u32 {
        if let Some(o) = Self::off(addr) {
            self.put16(o & !1, val);
        }
        BUS_OK
    }
    fn write32(&self, addr: u32, val: u32) -> u32 {
        if let Some(o) = Self::off(addr & !3) {
            self.put16(o, (val >> 16) as u16);
            if self.s.lock().mode == FlashMode::Read {
                self.put16(o + 2, val as u16);
            }
        }
        BUS_OK
    }
    fn write8(&self, addr: u32, val: u8) -> u32 {
        BusDevice::write16(self, addr & !1, val as u16 * 0x101)
    }
    fn write64(&self, addr: u32, val: u64) -> u32 {
        self.write32(addr & !7, (val >> 32) as u32)
    }
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, _addr: u32) -> *const AtomicU64 {
        &self.gen
    }
}

// --- the bus device ----------------------------------------------------

/// Where an access lands.
enum Target {
    HeartPiu(u32),
    HeartWidget(u32),
    Xbow(u32),
    Bridge(u32),
    None,
}

fn route(addr: u32) -> Target {
    if (HEART_PIU_BASE..HEART_PIU_END).contains(&addr) {
        return Target::HeartPiu(addr - HEART_PIU_BASE);
    }
    if (WIDGET_SPACE_BASE..WIDGET_SPACE_END).contains(&addr) {
        let w = (addr - WIDGET_SPACE_BASE) / WIDGET_SIZE;
        let off = (addr - WIDGET_SPACE_BASE) % WIDGET_SIZE;
        return match w {
            XBOW_WIDGET => Target::Xbow(off),
            HEART_WIDGET => Target::HeartWidget(off),
            BRIDGE_WIDGET => Target::Bridge(off),
            _ => Target::None,
        };
    }
    Target::None
}

pub struct Ip30 {
    pub heart: Heart,
    pub flash: Flash,
    xbow: Xbow,
    bridge: Bridge,
    trace: bool,
    seen: Mutex<HashSet<(u32, bool)>>,
    timer_on: AtomicBool,
}

impl Ip30 {
    /// `console` is the IOC3's UART A, the PROM's and IRIX's serial
    /// console; `aux` is UART B.
    pub fn new(
        banks_mb: [u32; 4],
        flash: Flash,
        rtc: Ds1687,
        mac: [u8; 6],
        console: Arc<dyn SerialBackend>,
        aux: Arc<dyn SerialBackend>,
    ) -> Self {
        Ip30 {
            heart: Heart::new(banks_mb),
            flash,
            xbow: Xbow::new(&[HEART_WIDGET, BRIDGE_WIDGET]),
            bridge: Bridge::new(rtc, mac, console, aux),
            trace: std::env::var_os("IRIS_IP30_TRACE").is_some(),
            seen: Mutex::new(HashSet::new()),
            timer_on: AtomicBool::new(false),
        }
    }

    /// Where HEART's interrupt levels go: the CPU's interrupt word.
    pub fn set_interrupts(&self, p: *const AtomicU64) {
        self.heart.cpu_irq.store(p as *mut AtomicU64, Ordering::Release);
    }

    /// Called with the new bank map on every MEMCFG write.
    pub fn set_memcfg_callback(&self, cb: Box<dyn Fn([Option<(u32, u32, u32)>; 4]) + Send + Sync>) {
        *self.heart.memcfg_cb.lock() = Some(cb);
    }

    /// Check HEART's compare every 100 µs. (A one-shot per compare write
    /// would be tighter; this is enough for the PROM.)
    pub fn set_timer_manager(self: &Arc<Self>, tm: Arc<TimerManager>) {
        if self.timer_on.swap(true, Ordering::AcqRel) {
            return;
        }
        let me = Arc::downgrade(self);
        let period = Duration::from_micros(100);
        tm.add_recurring(Instant::now() + period, period, (), move |_| match me.upgrade() {
            Some(ip30) => {
                ip30.heart.poll_timer();
                TimerReturn::Continue
            }
            None => TimerReturn::Delete,
        });
    }

    fn note(&self, addr: u32, write: bool, size: u32, val: u64) {
        let memcfg = (HEART_PIU_BASE + H_MEMCFG0..HEART_PIU_BASE + H_MEMCFG0 + 0x20).contains(&addr);
        if self.trace && (memcfg || self.seen.lock().insert((addr & !7, write))) {
            eprintln!("ip30: {} {:08x} size {} {:#x}", if write { "W" } else { "R" }, addr, size, val);
        }
    }

    fn read(&self, addr: u32, size: u32) -> Option<u64> {
        let v = match route(addr) {
            Target::HeartPiu(off) => {
                let v = self.heart.piu_read(off & !7);
                Some(narrow(v, addr, size))
            }
            Target::HeartWidget(off) => Some(narrow(self.heart.widget_read(off), addr, size)),
            Target::Xbow(off) => Some(narrow32(self.xbow.read32(off & !3), addr, size)),
            Target::Bridge(off) => self.bridge_read(off, size),
            Target::None => None,
        };
        self.note(addr, false, size, v.unwrap_or(u64::MAX));
        v
    }

    fn write(&self, addr: u32, size: u32, val: u64) -> bool {
        self.note(addr, true, size, val);
        match route(addr) {
            Target::HeartPiu(off) => {
                let old = if size == 8 { 0 } else { self.heart.piu_read(off & !7) };
                self.heart.piu_write(off & !7, widen(old, addr, size, val));
            }
            Target::HeartWidget(off) => {
                let old = self.heart.widget_read(off);
                self.heart.widget_write(off, widen(old, addr, size, val));
            }
            Target::Xbow(off) => {
                if size == 8 {
                    self.xbow.write32(off & !7 | 4, val as u32);
                } else {
                    self.xbow.write32(off & !3, val as u32);
                }
            }
            Target::Bridge(off) => return self.bridge_write(off, size, val),
            Target::None => return false,
        }
        true
    }

    fn bridge_read(&self, off: u32, size: u32) -> Option<u64> {
        if (BRIDGE_CONFIG_BASE..BRIDGE_CONFIG_END).contains(&off) {
            // Config space byte lanes are swizzled so that an aligned
            // dword reads as the PCI dword, and the bytes in it are in
            // big-endian order: PCI byte `where` is at `where ^ 3` (Linux's
            // `where ^ (4 - size)`).
            if size == 8 {
                let hi = self.bridge.config_read(off & !7) as u64;
                return Some(hi << 32 | self.bridge.config_read((off & !7) + 4) as u64);
            }
            return Some(narrow32(self.bridge.config_read(off & !3), off, size));
        }
        if (BRIDGE_DEVIO0..BRIDGE_DEVIO_END).contains(&off) {
            let (slot, o) = devio_slot(off)?;
            if slot != IOC3_SLOT {
                return None;
            }
            let mut ioc3 = self.bridge.ioc3.lock();
            return Some(match size {
                1 => ioc3.read8(o) as u64,
                4 => ioc3.read32(o) as u64,
                2 => (ioc3.read32(o) >> (8 * (2 - (o & 2)))) as u16 as u64,
                _ => ((ioc3.read32(o) as u64) << 32) | ioc3.read32(o + 4) as u64,
            });
        }
        if off < BRIDGE_CONFIG_BASE {
            return Some(match size {
                8 => self.bridge.reg_read32(off | 4) as u64,
                _ => narrow32(self.bridge.reg_read32(off & !3), off, size),
            });
        }
        match off {
            // Type 1 configuration: there is no bus behind a bridge here.
            0x2_8000..=0x2_ffff => Some(u64::MAX >> (64 - 8 * size)),
            // PCI interrupt acknowledge, and the external ATE SSRAM, which
            // this board does not have (it reads as zeros).
            0x3_0000..=0x3_ffff | 0x8_0000..=0xf_ffff => Some(0),
            // The second flash select (FLASH_MEM_ALT_BASE): no part fitted.
            0xe0_0000..=0xef_ffff => Some(u64::MAX >> (64 - 8 * size)),
            _ => None,
        }
    }

    fn bridge_write(&self, off: u32, size: u32, val: u64) -> bool {
        if (BRIDGE_CONFIG_BASE..BRIDGE_CONFIG_END).contains(&off) {
            let d = self.bridge.config_read(off & !3) as u64;
            let v = if size >= 4 { val } else { widen(d << 32, off & 3, size, val) >> 32 };
            self.bridge.config_write(off & !3, v as u32);
            return true;
        }
        if (BRIDGE_DEVIO0..BRIDGE_DEVIO_END).contains(&off) {
            let Some((slot, o)) = devio_slot(off) else { return false };
            if slot != IOC3_SLOT {
                return false;
            }
            let mut ioc3 = self.bridge.ioc3.lock();
            match size {
                1 => ioc3.write8(o, val as u8),
                8 => {
                    ioc3.write32(o, (val >> 32) as u32);
                    ioc3.write32(o + 4, val as u32);
                }
                _ => ioc3.write32(o, val as u32),
            }
            return true;
        }
        if off < BRIDGE_CONFIG_BASE {
            self.bridge.reg_write32(if size == 8 { off | 4 } else { off & !3 }, val as u32);
            return true;
        }
        matches!(off, 0x2_8000..=0x3_ffff | 0x8_0000..=0xf_ffff | 0xe0_0000..=0xef_ffff)
    }
}

/// The `size` bytes at `addr` of the big-endian doubleword `v`.
fn narrow(v: u64, addr: u32, size: u32) -> u64 {
    if size == 8 {
        return v;
    }
    let sh = 8 * (8 - size - (addr & 7 & !(size - 1)));
    (v >> sh) & ((1u64 << (8 * size)) - 1)
}

/// Same for a 32-bit register.
fn narrow32(v: u32, addr: u32, size: u32) -> u64 {
    if size >= 4 {
        return v as u64;
    }
    let sh = 8 * (4 - size - (addr & 3 & !(size - 1)));
    ((v >> sh) as u64) & ((1u64 << (8 * size)) - 1)
}

/// `old` with the `size` bytes at `addr` replaced by `val`.
fn widen(old: u64, addr: u32, size: u32, val: u64) -> u64 {
    if size == 8 {
        return val;
    }
    let sh = 8 * (8 - size - (addr & 7 & !(size - 1)));
    let m = ((1u64 << (8 * size)) - 1) << sh;
    (old & !m) | ((val << sh) & m)
}

impl BusDevice for Ip30 {
    fn read8(&self, addr: u32) -> BusRead8 {
        self.read(addr, 1).map_or(BusRead8::err(), |v| BusRead8::ok(v as u8))
    }
    fn read16(&self, addr: u32) -> BusRead16 {
        self.read(addr, 2).map_or(BusRead16::err(), |v| BusRead16::ok(v as u16))
    }
    fn read32(&self, addr: u32) -> BusRead32 {
        self.read(addr, 4).map_or(BusRead32::err(), |v| BusRead32::ok(v as u32))
    }
    fn read64(&self, addr: u32) -> BusRead64 {
        self.read(addr, 8).map_or(BusRead64::err(), BusRead64::ok)
    }
    fn write8(&self, addr: u32, val: u8) -> u32 {
        if self.write(addr, 1, val as u64) { BUS_OK } else { BUS_ERR }
    }
    fn write16(&self, addr: u32, val: u16) -> u32 {
        if self.write(addr, 2, val as u64) { BUS_OK } else { BUS_ERR }
    }
    fn write32(&self, addr: u32, val: u32) -> u32 {
        if self.write(addr, 4, val as u64) { BUS_OK } else { BUS_ERR }
    }
    fn write64(&self, addr: u32, val: u64) -> u32 {
        if self.write(addr, 8, val) { BUS_OK } else { BUS_ERR }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NullBackend;
    impl SerialBackend for NullBackend {
        fn send_byte(&self, _b: u8) {}
        fn recv_byte(&self) -> std::io::Result<u8> {
            Err(std::io::ErrorKind::WouldBlock.into())
        }
    }

    fn ip30() -> Ip30 {
        Ip30::new(
            [1024, 0, 0, 0],
            Flash::new(&[0xff; 16], None),
            Ds1687::new(None),
            [8, 0, 0x69, 1, 2, 3],
            Arc::new(NullBackend),
            Arc::new(NullBackend),
        )
    }

    #[test]
    fn memcfg_describes_the_configured_banks() {
        let m = ip30().heart.bank_map();
        assert_eq!(m[0], Some((0x2000_0000, 0x3fff_ffff, 0x4000_0000)));
        assert_eq!(m[1], None);
    }

    #[test]
    fn a_small_dimm_repeats_through_its_memcfg_window() {
        // The PROM's sizing probe: valid, 64 units (2 GB), at 2 GB up.
        let m = decode_memcfg(&[0x81bf_0040, 0, 0, 0], &[256, 0, 0, 0]);
        assert_eq!(m[0], Some((0xa000_0000, 0x0fff_ffff, 0x6000_0000)));
        // Placed past 4 GB: not reachable on a 32-bit bus, so left out.
        let m = decode_memcfg(&[0x819f_0080, 0, 0, 0], &[1024, 0, 0, 0]);
        assert_eq!(m[0], None);
    }

    #[test]
    fn heart_count_runs_at_12_5_mhz() {
        let h = ip30();
        let a = h.read64(0x0ff2_0000).data;
        std::thread::sleep(Duration::from_millis(20));
        let b = h.read64(0x0ff2_0000).data;
        let ticks = b - a;
        assert!((200_000..400_000).contains(&ticks), "{ticks} ticks in 20 ms");
    }

    #[test]
    fn heart_levels_drive_the_cause_bits() {
        let h = ip30();
        let word = AtomicU64::new(1 << 15); // IP7, the CPU's own: untouched
        h.set_interrupts(&word);
        h.write64(0x0ff1_0000, 1 << 50 | 1 << 3); // IMR0: timer, vector 3
        h.write64(0x0ff1_0020, 1 << 3); // SET_ISR vector 3: level 0
        assert_eq!(word.load(Ordering::SeqCst), 1 << 15 | 1 << 10);
        h.write64(0x0ff1_0020, 1 << 50); // the timer: level 3
        assert_eq!(word.load(Ordering::SeqCst), 1 << 15 | 1 << 10 | 1 << 13);
        h.write64(0x0ff1_0028, 1 << 3 | 1 << 50); // CLR_ISR both
        assert_eq!(word.load(Ordering::SeqCst), 1 << 15);
    }

    #[test]
    fn flash_answers_id_status_program_and_erase() {
        let f = Flash::new(&[0x12, 0x34, 0x56, 0x78], None);
        assert_eq!(f.read32(FLASH_BASE).data, 0x1234_5678);
        f.write16(FLASH_BASE, 0x9090);
        assert_eq!(f.read16(FLASH_BASE).data, 0x00b0);
        assert_eq!(f.read16(FLASH_BASE + 2).data, 0x66a8);
        f.write16(FLASH_BASE, 0x4040);
        f.write16(FLASH_BASE + 2, 0x0f0f); // program: AND into 0x5678
        assert_eq!(f.read16(FLASH_BASE).data & 0x80, 0x80);
        f.write16(FLASH_BASE, 0xffff);
        assert_eq!(f.read16(FLASH_BASE + 2).data, 0x0608);
        f.write16(FLASH_BASE, 0x2020);
        f.write16(FLASH_BASE, 0xd0d0);
        f.write16(FLASH_BASE, 0xffff);
        assert_eq!(f.read32(FLASH_BASE).data, 0xffff_ffff);
    }

    #[test]
    fn rtc_nvram_and_extended_ram_hold_what_is_written() {
        let h = ip30();
        let rtc = |r: u8, v: Option<u8>| {
            h.write8(0x1f6a_0000, r);
            match v {
                Some(v) => {
                    h.write8(0x1f6c_0000, v);
                    0
                }
                None => h.read8(0x1f6c_0000).data,
            }
        };
        rtc(0x0e, Some(0x5a));
        assert_eq!(rtc(0x0e, None), 0x5a);
        assert_eq!(rtc(0x0d, None), 0x80); // valid RAM and time
        rtc(0x0a, Some(0x30)); // bank 1
        rtc(0x50, Some(0x07));
        rtc(0x53, Some(0xa5));
        rtc(0x50, Some(0x07));
        assert_eq!(rtc(0x53, None), 0xa5);
        assert!(rtc(0x00, None) < 0x60); // BCD seconds
    }

    /// The 1-wire protocol, the way sgi_w1.c drives it.
    struct Master<'a>(&'a mut OneWire);
    impl Master<'_> {
        fn reset(&mut self) -> bool {
            self.0.write(520 << 10 | 65 << 2);
            self.0.read() & MCR_RD_DATA == 0
        }
        fn bit(&mut self, b: bool) -> bool {
            self.0.write(if b { 6 << 10 | 13 << 2 } else { 80 << 10 | 30 << 2 });
            self.0.read() & MCR_RD_DATA != 0
        }
        fn byte(&mut self, b: u8) -> u8 {
            (0..8).fold(0, |acc, i| acc | (self.bit(b >> i & 1 != 0) as u8) << i)
        }
    }

    #[test]
    fn the_mac_nic_answers_read_rom_and_read_memory() {
        let mac = [0x08, 0x00, 0x69, 0x0b, 0x7e, 0xba];
        let mut w = OneWire::with(Ds2502::mac_nic(mac));
        let mut m = Master(&mut w);
        assert!(m.reset(), "presence pulse");
        m.byte(0x33);
        let rom: Vec<u8> = (0..8).map(|_| m.byte(0xff)).collect();
        assert_eq!(rom[0], 0x91);
        assert_eq!(rom[7], crc8(&rom[..7]));
        // Linux's ioc3_get_eaddr_nic: match ROM, read memory from 0.
        assert!(m.reset());
        m.byte(0x55);
        for &b in &rom {
            m.byte(b);
        }
        m.byte(0xf0);
        m.byte(0);
        m.byte(0);
        let mut nic = [0u8; 14];
        for i in (0..14).rev() {
            nic[i] = m.byte(0xff);
        }
        assert_eq!(nic[13], crc8(&[0xf0, 0, 0]));
        assert_eq!(&nic[2..8], &mac);
        // The page's CRC16, inverted and low byte first, closes it.
        let page: Vec<u8> = (2..=12).rev().map(|i| nic[i]).collect();
        assert_eq!(!crc16(&page), u16::from_le_bytes([nic[1], nic[0]]));
    }

    #[test]
    fn search_rom_finds_the_one_part() {
        let mut w = OneWire::with(Ds2502::mac_nic([1, 2, 3, 4, 5, 6]));
        let mut m = Master(&mut w);
        assert!(m.reset());
        m.byte(0xf0);
        let mut rom = 0u64;
        for i in 0..64 {
            let b = m.bit(true);
            let nb = m.bit(true);
            assert_ne!(b, nb, "one part: no discrepancy at bit {i}");
            m.bit(b);
            rom |= (b as u64) << i;
        }
        assert_eq!(rom as u8, 0x91);
        // No part on the Bridge: no presence.
        let mut empty = OneWire::default();
        assert!(!Master(&mut empty).reset());
    }

    #[test]
    fn widget_ids_identify_the_parts() {
        let h = ip30();
        let part = |a| (h.read32(a).data >> 12) & 0xffff;
        assert_eq!(part(0x1000_0004), 0x0000); // Xbow
        assert_eq!(part(0x1800_0004), 0xc001); // HEART
        assert_eq!(part(0x1f00_0004), 0xc002); // Bridge
        assert!(h.read32(0x1100_0004).status != BUS_OK); // nothing on widget 1
    }

    #[test]
    fn xbow_reports_the_live_links() {
        let h = ip30();
        let status = |port: u32| h.read32(0x1000_0000 + 0x100 + (port & 7) * 0x40 + 0x14).data;
        assert_eq!(status(8), XB_STAT_LINKALIVE);
        assert_eq!(status(0xf), XB_STAT_LINKALIVE);
        assert_eq!(status(0xc), 0);
    }

    #[test]
    fn pci_config_finds_the_ioc3_and_sizes_its_bar() {
        let h = ip30();
        let cfg = |slot: u32, reg: u32| 0x1f02_0000 + slot * 0x1000 + reg;
        assert_eq!(h.read32(cfg(2, 0)).data, 0x0003_10a9);
        assert_eq!(h.read32(cfg(5, 0)).data, 0xffff_ffff);
        h.write32(cfg(2, 0x10), 0xffff_ffff);
        assert_eq!(h.read32(cfg(2, 0x10)).data, 0xfff0_0000);
        assert_eq!(h.read16(cfg(2, 0)).data, 0x0003); // device ID: big-endian lanes
        assert_eq!(h.read8(cfg(2, 3)).data, 0xa9); // PCI byte 0
    }
}
