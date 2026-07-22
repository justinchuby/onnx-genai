//! CPU hardware detection via pytorch/cpuinfo.
//!
//! Provides a safe Rust API over the cpuinfo C library for detecting:
//! - ISA features (AVX2, AVX-512, AMX, NEON, SVE, SME, ...)
//! - Cache hierarchy (L1/L2/L3 size, line size, associativity)
//! - Core topology (P/E cores, NUMA nodes, packages)
//! - Processor microarchitecture identification
//!
//! # Usage
//!
//! ```no_run
//! use onnx_runtime_cpuinfo::CpuInfo;
//!
//! let cpu = CpuInfo::detect().unwrap();
//! println!("Cores: {}", cpu.topology().physical_cores());
//! println!("AVX2: {}", cpu.isa().has_avx2());
//! println!("L2 cache: {} KB", cpu.cache_l2_size() / 1024);
//! ```

mod ffi;

use std::sync::Once;

static INIT: Once = Once::new();

/// Initialize cpuinfo. Called automatically on first use.
fn ensure_init() {
    INIT.call_once(|| {
        let ok = unsafe { ffi::cpuinfo_initialize() };
        assert!(ok, "cpuinfo_initialize() failed");
    });
}

/// CPU hardware information. Detected once, immutable thereafter.
#[derive(Debug)]
pub struct CpuInfo {
    isa: IsaFeatures,
    caches: Vec<CacheLevel>,
    topology: CpuTopology,
}

impl CpuInfo {
    /// Detect all CPU information. Safe to call multiple times (idempotent).
    pub fn detect() -> Result<Self, CpuInfoError> {
        ensure_init();
        Ok(Self {
            isa: IsaFeatures::detect(),
            caches: CacheLevel::detect_all(),
            topology: CpuTopology::detect(),
        })
    }

    pub fn isa(&self) -> &IsaFeatures {
        &self.isa
    }

    pub fn caches(&self) -> &[CacheLevel] {
        &self.caches
    }

    pub fn topology(&self) -> &CpuTopology {
        &self.topology
    }

    /// Convenience: L2 cache size in bytes (first L2 found, or 256KB default).
    pub fn cache_l2_size(&self) -> usize {
        self.caches
            .iter()
            .find(|c| c.level == 2)
            .map(|c| c.size_bytes)
            .unwrap_or(256 * 1024)
    }

    /// Convenience: L3 cache size in bytes (0 if none).
    pub fn cache_l3_size(&self) -> usize {
        self.caches
            .iter()
            .find(|c| c.level == 3)
            .map(|c| c.size_bytes)
            .unwrap_or(0)
    }
}

// =============================================================================
// ISA Features
// =============================================================================

/// Detected instruction set features.
#[derive(Debug, Clone, Default)]
pub struct IsaFeatures {
    // x86
    pub avx: bool,
    pub avx2: bool,
    pub fma3: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vl: bool,
    pub avx512vnni: bool,
    pub avx512bf16: bool,
    pub avx512fp16: bool,
    pub amx_int8: bool,
    pub amx_bf16: bool,
    pub amx_tile: bool,
    // ARM
    pub neon: bool,
    pub sve: bool,
    pub sve2: bool,
    pub dotprod: bool,
    pub fp16_arith: bool,
    pub bf16: bool,
    pub i8mm: bool,
    pub sme: bool,
}

impl IsaFeatures {
    fn detect() -> Self {
        let mut f = Self::default();

        #[cfg(target_arch = "x86_64")]
        {
            f.avx = unsafe { ffi::cpuinfo_has_x86_avx() };
            f.avx2 = unsafe { ffi::cpuinfo_has_x86_avx2() };
            f.fma3 = unsafe { ffi::cpuinfo_has_x86_fma3() };
            f.avx512f = unsafe { ffi::cpuinfo_has_x86_avx512f() };
            f.avx512bw = unsafe { ffi::cpuinfo_has_x86_avx512bw() };
            f.avx512vl = unsafe { ffi::cpuinfo_has_x86_avx512vl() };
            f.avx512vnni = unsafe { ffi::cpuinfo_has_x86_avx512vnni() };
            f.avx512bf16 = unsafe { ffi::cpuinfo_has_x86_avx512bf16() };
            // AMX detection — cpuinfo may not have these yet, gate with cfg
            // f.amx_int8 = unsafe { cpuinfo_has_x86_amx_int8() };
            // f.amx_bf16 = unsafe { cpuinfo_has_x86_amx_bf16() };
        }

        #[cfg(target_arch = "aarch64")]
        {
            f.neon = true; // always available on aarch64
            f.dotprod = unsafe { ffi::cpuinfo_has_arm_neon_dot() };
            f.fp16_arith = unsafe { ffi::cpuinfo_has_arm_neon_fp16_arith() };
            f.bf16 = unsafe { ffi::cpuinfo_has_arm_bf16() };
            f.i8mm = unsafe { ffi::cpuinfo_has_arm_i8mm() };
            f.sve = unsafe { ffi::cpuinfo_has_arm_sve() };
            f.sve2 = unsafe { ffi::cpuinfo_has_arm_sve2() };
            // SME not yet in cpuinfo — fall back to std::arch detection
        }

        f
    }

    // Convenience methods

    pub fn has_avx2(&self) -> bool {
        self.avx2
    }

    pub fn has_avx512(&self) -> bool {
        self.avx512f
    }

    pub fn has_vnni(&self) -> bool {
        self.avx512vnni
    }

    pub fn has_amx(&self) -> bool {
        self.amx_int8 || self.amx_bf16
    }

    pub fn has_neon(&self) -> bool {
        self.neon
    }

    pub fn has_sve(&self) -> bool {
        self.sve
    }

    /// Best GEMM strategy for this CPU.
    pub fn best_gemm_strategy(&self) -> GemmStrategy {
        if self.amx_int8 || self.amx_bf16 {
            GemmStrategy::Amx
        } else if self.avx512vnni {
            GemmStrategy::Avx512Vnni
        } else if self.avx512f {
            GemmStrategy::Avx512
        } else if self.avx2 && self.fma3 {
            GemmStrategy::Avx2Fma
        } else if self.sve {
            GemmStrategy::Sve
        } else if self.neon && self.dotprod {
            GemmStrategy::NeonDotprod
        } else if self.neon {
            GemmStrategy::Neon
        } else {
            GemmStrategy::Generic
        }
    }
}

/// Kernel dispatch strategy for GEMM operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GemmStrategy {
    Amx,
    Avx512Vnni,
    Avx512,
    Avx2Fma,
    Sve,
    NeonDotprod,
    Neon,
    Generic,
}

// =============================================================================
// Cache Hierarchy
// =============================================================================

/// One level of the cache hierarchy.
#[derive(Debug, Clone)]
pub struct CacheLevel {
    pub level: u8,
    pub size_bytes: usize,
    pub line_size: usize,
    pub associativity: u32,
    pub processor_count: u32, // how many logical processors share this cache
}

impl CacheLevel {
    fn detect_all() -> Vec<Self> {
        let mut caches = Vec::new();

        // L1 data cache
        let count = unsafe { ffi::cpuinfo_get_l1d_caches_count() };
        if count > 0 {
            let ptr = unsafe { ffi::cpuinfo_get_l1d_caches() };
            if !ptr.is_null() {
                let c = unsafe { &*ptr };
                caches.push(CacheLevel {
                    level: 1,
                    size_bytes: c.size as usize,
                    line_size: c.line_size as usize,
                    associativity: c.associativity,
                    processor_count: c.processor_count,
                });
            }
        }

        // L2 cache
        let count = unsafe { ffi::cpuinfo_get_l2_caches_count() };
        if count > 0 {
            let ptr = unsafe { ffi::cpuinfo_get_l2_caches() };
            if !ptr.is_null() {
                let c = unsafe { &*ptr };
                caches.push(CacheLevel {
                    level: 2,
                    size_bytes: c.size as usize,
                    line_size: c.line_size as usize,
                    associativity: c.associativity,
                    processor_count: c.processor_count,
                });
            }
        }

        // L3 cache
        let count = unsafe { ffi::cpuinfo_get_l3_caches_count() };
        if count > 0 {
            let ptr = unsafe { ffi::cpuinfo_get_l3_caches() };
            if !ptr.is_null() {
                let c = unsafe { &*ptr };
                caches.push(CacheLevel {
                    level: 3,
                    size_bytes: c.size as usize,
                    line_size: c.line_size as usize,
                    associativity: c.associativity,
                    processor_count: c.processor_count,
                });
            }
        }

        caches
    }
}

// =============================================================================
// Topology
// =============================================================================

/// CPU core topology.
#[derive(Debug, Clone)]
pub struct CpuTopology {
    physical_cores: usize,
    logical_processors: usize,
    packages: usize,
    cores: Vec<CoreInfo>,
}

impl CpuTopology {
    fn detect() -> Self {
        let processors = unsafe { ffi::cpuinfo_get_processors_count() } as usize;
        let cores_count = unsafe { ffi::cpuinfo_get_cores_count() } as usize;
        let packages = unsafe { ffi::cpuinfo_get_packages_count() } as usize;

        let mut cores = Vec::with_capacity(cores_count);
        let cores_ptr = unsafe { ffi::cpuinfo_get_cores() };

        if !cores_ptr.is_null() {
            for i in 0..cores_count {
                let c = unsafe { &*cores_ptr.add(i) };
                cores.push(CoreInfo {
                    core_id: i,
                    processor_start: c.processor_start as usize,
                    processor_count: c.processor_count as usize,
                    core_type: CoreType::from_cpuinfo(c),
                    frequency: c.frequency,
                });
            }
        }

        Self {
            physical_cores: cores_count,
            logical_processors: processors,
            packages,
            cores,
        }
    }

    pub fn physical_cores(&self) -> usize {
        self.physical_cores
    }

    pub fn logical_processors(&self) -> usize {
        self.logical_processors
    }

    pub fn packages(&self) -> usize {
        self.packages
    }

    pub fn cores(&self) -> &[CoreInfo] {
        &self.cores
    }

    /// Number of Performance cores (or all cores if homogeneous).
    pub fn performance_cores(&self) -> usize {
        let p_count = self
            .cores
            .iter()
            .filter(|c| c.core_type == CoreType::Performance)
            .count();
        if p_count == 0 {
            self.physical_cores
        } else {
            p_count
        }
    }

    /// Number of Efficiency cores (0 if homogeneous).
    pub fn efficiency_cores(&self) -> usize {
        self.cores
            .iter()
            .filter(|c| c.core_type == CoreType::Efficiency)
            .count()
    }
}

/// Information about a single core.
#[derive(Debug, Clone)]
pub struct CoreInfo {
    pub core_id: usize,
    pub processor_start: usize,
    pub processor_count: usize,
    pub core_type: CoreType,
    /// Max frequency in Hz (0 if unknown).
    pub frequency: u64,
}

/// Core type classification for hybrid architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoreType {
    Performance,
    Efficiency,
    Unknown,
}

impl CoreType {
    fn from_cpuinfo(_core: &ffi::cpuinfo_core) -> Self {
        // cpuinfo exposes core.frequency — higher freq = P-core heuristic.
        // On Intel hybrid, cpuinfo sets cpuinfo_uarch for each core.
        // For now: use Unknown, refine when we can read uarch per-core.
        // TODO: check core.uarch for Raptor Cove (P) vs Gracemont (E)
        Self::Unknown
    }
}

// =============================================================================
// Optimal Configuration
// =============================================================================

/// Compute optimal tiling config based on detected cache.
#[derive(Debug, Clone)]
pub struct TileConfig {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
}

impl TileConfig {
    /// Choose tile sizes so working set fits in L2 cache.
    /// For fp32 GEMM: A_tile[M,K] + B_tile[K,N] + C_tile[M,N] ≤ 75% of L2.
    pub fn for_l2(l2_bytes: usize) -> Self {
        let budget = l2_bytes * 3 / 4;
        let tile_k: usize = 256;

        // Solve: (M*K + K*N + M*N) * 4 ≤ budget, with M ≈ N
        let max_mn = ((budget / 4) as f64 / (2.0 * tile_k as f64 + 1.0)).sqrt() as usize;
        let tile_m = max_mn.clamp(32, 512).next_power_of_two();
        let tile_n = tile_m;

        Self {
            tile_m,
            tile_n,
            tile_k,
        }
    }
}

/// Recommended thread pool configuration based on topology.
#[derive(Debug, Clone)]
pub struct ThreadPoolRecommendation {
    /// Number of threads for compute.
    pub num_threads: usize,
    /// Whether to pin threads to cores.
    pub pin_to_cores: bool,
    /// Use only P-cores for latency-sensitive work.
    pub prefer_p_cores: bool,
}

impl ThreadPoolRecommendation {
    pub fn from_topology(topo: &CpuTopology) -> Self {
        let p_cores = topo.performance_cores();
        Self {
            // Use P-cores count for compute threads (no HT)
            num_threads: p_cores,
            pin_to_cores: true,
            prefer_p_cores: topo.efficiency_cores() > 0,
        }
    }
}

// =============================================================================
// Error
// =============================================================================

#[derive(Debug, thiserror::Error)]
pub enum CpuInfoError {
    #[error("cpuinfo initialization failed")]
    InitFailed,
}
