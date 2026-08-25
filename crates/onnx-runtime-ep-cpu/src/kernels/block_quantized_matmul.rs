//! Correctness-first `BlockQuantizedMatMul` for native GGUF block formats.
//!
//! The packed weight tensor keeps llama.cpp's serialized block layout. MXFP4
//! decoding follows OCP MX E2M1/E8M0 and llama.cpp's `block_mxfp4`; IQ
//! decoding follows llama.cpp's native super-block layouts and audited grids.
//! This CPU kernel is a memory-format baseline: it dequantizes `packed_B` to a
//! dense f32 matrix, caches that f32 expansion only for constant weights, and
//! runs dense GEMM. It does not perform quantized-domain matmul compute.

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use onnx_runtime_ep_api::{
    EpError, Kernel, KernelFactory, Result, TensorBacking, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node};
use onnx_runtime_quantization::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XS_SIGNS, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID,
};
use rayon::prelude::*;

use super::block_dequant::{decode_e2m1, decode_e8m0_scale};
use super::matmul::gemm;
use super::{check_arity, to_dense_bytes};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};
use crate::strided::numel;

const OP: &str = "BlockQuantizedMatMul";
const DOMAIN: &str = onnx_runtime_ir::RUNTIME_DOMAIN;
const LAYOUT_VERSION: i64 = 1;

const MXFP4_QK: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 17;
const IQ4_NL_QK: usize = 32;
const IQ4_NL_BLOCK_BYTES: usize = 18;
const IQ_SUPER_QK: usize = 256;
const IQ4_XS_BLOCK_BYTES: usize = 136;
const IQ3_S_BLOCK_BYTES: usize = 110;
const IQ3_XXS_BLOCK_BYTES: usize = 98;
const IQ2_S_BLOCK_BYTES: usize = 82;
const IQ2_XS_BLOCK_BYTES: usize = 74;
const IQ2_XXS_BLOCK_BYTES: usize = 66;
const IQ1_S_BLOCK_BYTES: usize = 50;
const IQ1_M_BLOCK_BYTES: usize = 56;
const IQ1_S_DELTA: f32 = 0.125;
const IQ1_M_DELTA: f32 = 0.125;
pub(super) const DEFAULT_DENSE_WEIGHT_CACHE_BYTES: usize = 256 * 1024 * 1024;
const DENSE_WEIGHT_CACHE_BYTES_ENV: &str = "ONNX_GENAI_CPU_BLOCK_QUANT_CACHE_BYTES";

pub static BLOCK_QUANT_MATMUL_CACHED_DENSE_TEST_HITS: AtomicUsize = AtomicUsize::new(0);
pub static BLOCK_QUANT_MATMUL_DENSE_EXPANSIONS: AtomicUsize = AtomicUsize::new(0);

#[cfg(test)]
static BLOCK_QUANTIZED_MATMUL_DENSE_F32_TEST_HITS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

// SIMD decoder uses doubled E2M1 integers with half the shared E8M0 scale.
#[cfg(target_arch = "x86_64")]
const E2M1_DOUBLED: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// llama.cpp commit b15ca938, ggml-common.h::kvalues_iq4nl.
const IQ4_NL_CODEBOOK: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

// Vendored byte-for-byte from llama.cpp commit b15ca938, ggml-common.h.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum BlockFormat {
    Mxfp4,
    Iq4Nl,
    Iq4Xs,
    Iq3S,
    Iq3Xxs,
    Iq2S,
    Iq2Xs,
    Iq2Xxs,
    Iq1S,
    Iq1M,
}

impl BlockFormat {
    pub(super) fn parse(value: &str) -> Result<Self> {
        match value {
            "mxfp4" => Ok(Self::Mxfp4),
            "iq4_nl" => Ok(Self::Iq4Nl),
            "iq4_xs" => Ok(Self::Iq4Xs),
            "iq3_s" => Ok(Self::Iq3S),
            "iq3_xxs" => Ok(Self::Iq3Xxs),
            "iq2_s" => Ok(Self::Iq2S),
            "iq2_xs" => Ok(Self::Iq2Xs),
            "iq2_xxs" => Ok(Self::Iq2Xxs),
            "iq1_s" => Ok(Self::Iq1S),
            "iq1_m" => Ok(Self::Iq1M),
            _ => Err(error(format!(
                "unsupported format '{value}'; supported formats are mxfp4, iq4_nl, iq4_xs, iq3_s, iq3_xxs, iq2_s, iq2_xs, iq2_xxs, iq1_s, and iq1_m"
            ))),
        }
    }

    pub(super) fn qk(self) -> usize {
        match self {
            Self::Mxfp4 => MXFP4_QK,
            Self::Iq4Nl => IQ4_NL_QK,
            Self::Iq4Xs
            | Self::Iq3S
            | Self::Iq3Xxs
            | Self::Iq2S
            | Self::Iq2Xs
            | Self::Iq2Xxs
            | Self::Iq1S
            | Self::Iq1M => IQ_SUPER_QK,
        }
    }

    pub(super) fn block_bytes(self) -> usize {
        match self {
            Self::Mxfp4 => MXFP4_BLOCK_BYTES,
            Self::Iq4Nl => IQ4_NL_BLOCK_BYTES,
            Self::Iq4Xs => IQ4_XS_BLOCK_BYTES,
            Self::Iq3S => IQ3_S_BLOCK_BYTES,
            Self::Iq3Xxs => IQ3_XXS_BLOCK_BYTES,
            Self::Iq2S => IQ2_S_BLOCK_BYTES,
            Self::Iq2Xs => IQ2_XS_BLOCK_BYTES,
            Self::Iq2Xxs => IQ2_XXS_BLOCK_BYTES,
            Self::Iq1S => IQ1_S_BLOCK_BYTES,
            Self::Iq1M => IQ1_M_BLOCK_BYTES,
        }
    }

    fn scalar_decoder(self) -> fn(&[u8], &mut [f32]) {
        match self {
            Self::Mxfp4 => decode_mxfp4_block,
            Self::Iq4Nl => decode_iq4_nl_block,
            Self::Iq4Xs => decode_iq4_xs_block,
            Self::Iq3S => decode_iq3_s_block,
            Self::Iq3Xxs => decode_iq3_xxs_block,
            Self::Iq2S => decode_iq2_s_block,
            Self::Iq2Xs => decode_iq2_xs_block,
            Self::Iq2Xxs => decode_iq2_xxs_block,
            Self::Iq1S => decode_iq1_s_block,
            Self::Iq1M => decode_iq1_m_block,
        }
    }

    fn decoder(self) -> fn(&[u8], &mut [f32]) {
        #[cfg(target_arch = "x86_64")]
        if std::arch::is_x86_feature_detected!("avx2") {
            return match self {
                Self::Mxfp4 => decode_mxfp4_block_avx2_dispatch,
                Self::Iq4Nl => decode_iq4_nl_block_avx2_dispatch,
                Self::Iq4Xs => decode_iq4_xs_block_avx2_dispatch,
                _ => self.scalar_decoder(),
            };
        }
        self.scalar_decoder()
    }
}

pub struct BlockQuantizedMatMulKernel {
    k: usize,
    n: usize,
    format: BlockFormat,
    packed_b_constant: bool,
    weight_identity: DenseWeightIdentity,
    weight_cache: DenseWeightCache,
}

pub struct BlockQuantizedMatMulFactory;

impl KernelFactory for BlockQuantizedMatMulFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_positive_attr(node, "K")?;
        let n = required_positive_attr(node, "N")?;
        let layout_version = optional_int_attr(node, "block_layout_version")?.unwrap_or(1);
        if layout_version != LAYOUT_VERSION {
            return Err(error(format!(
                "block_layout_version must be {LAYOUT_VERSION}, got {layout_version}"
            )));
        }
        let format = match node.attr("format") {
            Some(attribute) => attribute
                .as_str()
                .ok_or_else(|| error("attribute 'format' must be a UTF-8 string"))
                .and_then(BlockFormat::parse)?,
            None => return Err(error("missing required string attribute 'format'")),
        };

        Ok(Box::new(BlockQuantizedMatMulKernel {
            k,
            n,
            format,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        }))
    }
}

impl Kernel for BlockQuantizedMatMulKernel {
    fn set_constant_inputs(&mut self, constant_inputs: &[bool]) {
        self.packed_b_constant = constant_inputs.get(1).copied().unwrap_or(false);
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 2, 3, 1)?;
        require_compute_dtype("A", inputs[0].dtype)?;
        require_dtype("packed_B", inputs[1].dtype, DataType::Uint8)?;
        require_compute_dtype("Y", outputs[0].dtype)?;
        if outputs[0].dtype != inputs[0].dtype {
            return Err(error(format!(
                "Y dtype {:?} must match A dtype {:?}",
                outputs[0].dtype, inputs[0].dtype
            )));
        }

        let a_shape = inputs[0].shape;
        if a_shape.is_empty() || a_shape[a_shape.len() - 1] != self.k {
            return Err(error(format!(
                "A must have rank >= 1 and last dimension K={}, got {:?}",
                self.k, a_shape
            )));
        }
        let expected_output_shape = [&a_shape[..a_shape.len() - 1], &[self.n]].concat();
        require_shape("Y", outputs[0].shape, &expected_output_shape)?;

        let blocks = self.k.div_ceil(self.format.qk());
        require_shape(
            "packed_B",
            inputs[1].shape,
            &[self.n, blocks, self.format.block_bytes()],
        )?;

        let bias = if let Some(bias) = inputs.get(2).filter(|input| !input.is_absent()) {
            require_compute_dtype("bias", bias.dtype)?;
            if bias.dtype != inputs[0].dtype {
                return Err(error(format!(
                    "bias dtype {:?} must match A dtype {:?}",
                    bias.dtype, inputs[0].dtype
                )));
            }
            require_shape("bias", bias.shape, &[self.n])?;
            Some(to_dense_f32_widen(OP, bias)?.into_owned())
        } else {
            None
        };

        let activations = to_dense_f32_widen(OP, &inputs[0])?;
        let owned_weight;
        let cached_weight;
        let weight_kn = if self.packed_b_constant {
            let resolved = self.weight_identity.resolve(
                &inputs[1],
                self.format,
                self.k,
                self.n,
                0,
                None,
                || packed_tensor_bytes(&inputs[1]),
            )?;
            let mut resolved_payload = resolved.payload;
            let (weight, status) =
                self.weight_cache
                    .get_or_insert_with(resolved.key.as_ref(), || {
                        BLOCK_QUANT_MATMUL_DENSE_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
                        let packed = match resolved_payload.take() {
                            Some(packed) => packed,
                            None => packed_tensor_bytes(&inputs[1])?,
                        };
                        dequantize_weight_kn(self.format, self.k, self.n, &packed)
                    })?;
            if matches!(status, DenseWeightCacheStatus::Hit) {
                BLOCK_QUANT_MATMUL_CACHED_DENSE_TEST_HITS.fetch_add(1, Ordering::Relaxed);
            }
            cached_weight = weight;
            cached_weight.as_slice()
        } else {
            BLOCK_QUANT_MATMUL_DENSE_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
            let packed = packed_tensor_bytes(&inputs[1])?;
            owned_weight = dequantize_weight_kn(self.format, self.k, self.n, &packed)?;
            &owned_weight
        };

        let m = numel(&a_shape[..a_shape.len() - 1]);
        let result_elements = m
            .checked_mul(self.n)
            .ok_or_else(|| error("Y element count overflow"))?;
        let mut result = vec![0.0f32; result_elements];
        #[cfg(test)]
        BLOCK_QUANTIZED_MATMUL_DENSE_F32_TEST_HITS
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        gemm(&activations, weight_kn, &mut result, m, self.k, self.n)?;
        if let Some(bias) = bias {
            for row in result.chunks_exact_mut(self.n) {
                for (value, bias) in row.iter_mut().zip(&bias) {
                    *value += bias;
                }
            }
        }
        write_dense_f32_narrow(OP, &mut outputs[0], &result)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn require_compute_dtype(name: &str, got: DataType) -> Result<()> {
    if matches!(
        got,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        Ok(())
    } else {
        Err(error(format!(
            "{name} has unsupported dtype {got:?}; BlockQuantizedMatMul computes floating-point \
             activations in Float32 and supports Float32, Float16, or BFloat16 storage. Cast \
             {name} to one of those dtypes before this operator"
        )))
    }
}

impl BlockQuantizedMatMulKernel {
    #[cfg(test)]
    fn dequantize_weight_kn(&self, packed: &TensorView) -> Result<Vec<f32>> {
        let packed = packed_tensor_bytes(packed)?;
        BLOCK_QUANT_MATMUL_DENSE_EXPANSIONS.fetch_add(1, Ordering::Relaxed);
        dequantize_weight_kn(self.format, self.k, self.n, &packed)
    }
}

/// Stable identity for one immutable packed initializer slot in one kernel.
///
/// The session owns kernels per `(session, node, resolved shapes)` and keeps
/// initializer storage alive for the session. `Kernel::set_constant_inputs`
/// promises that a marked slot never changes. Therefore an opaque pointer is
/// only an identity-change prefilter: a same-pointer hit is accepted because of
/// the constant-slot ownership contract, not because an address alone proves
/// identity. External mmap inputs additionally carry a process-unique mapping
/// id. On any observable source/layout change, the generation advances and all
/// memoized content keys are discarded before the new payload is hashed.
#[derive(Default)]
pub(super) struct DenseWeightIdentity {
    inner: Mutex<DenseWeightIdentityState>,
}

#[derive(Default)]
struct DenseWeightIdentityState {
    generation: u64,
    metadata: Option<DenseWeightTensorMetadata>,
    keys: HashMap<DenseWeightSubKey, Arc<DenseWeightCacheKey>>,
    #[cfg(test)]
    hashed_bytes: usize,
    #[cfg(test)]
    materialized_bytes: usize,
}

#[derive(Clone, Debug)]
struct DenseWeightTensorMetadata {
    dtype: DataType,
    format: BlockFormat,
    shape: Arc<[usize]>,
    strides: Arc<[i64]>,
    byte_len: usize,
    source: DenseWeightSource,
}

impl DenseWeightTensorMetadata {
    fn matches(&self, view: &TensorView, format: BlockFormat, source: DenseWeightSource) -> bool {
        self.dtype == view.dtype
            && self.format == format
            && self.shape.as_ref() == view.shape
            && self.strides.as_ref() == view.strides
            && self.byte_len == view.byte_size()
            && self.source == source
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct DenseWeightSubKey {
    role: u8,
    expert: Option<usize>,
    in_features: usize,
    out_features: usize,
}

pub(super) struct ResolvedDenseWeight<'a> {
    pub(super) key: Arc<DenseWeightCacheKey>,
    /// Present only when resolving a new content key. The caller can reuse this
    /// payload for the cache build, avoiding a second strided materialization.
    pub(super) payload: Option<Cow<'a, [u8]>>,
}

impl DenseWeightIdentity {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn resolve<'a>(
        &self,
        view: &'a TensorView<'_>,
        format: BlockFormat,
        in_features: usize,
        out_features: usize,
        role: u8,
        expert: Option<usize>,
        payload: impl FnOnce() -> Result<Cow<'a, [u8]>>,
    ) -> Result<ResolvedDenseWeight<'a>> {
        let source = DenseWeightSource::from_tensor(view)?;
        let mut inner = self
            .inner
            .lock()
            .expect("BlockQuantized weight identity lock poisoned");
        let identity_changed = inner
            .metadata
            .as_ref()
            .is_none_or(|metadata| !metadata.matches(view, format, source));
        if identity_changed {
            inner.generation = inner
                .generation
                .checked_add(1)
                .ok_or_else(|| error("constant packed weight identity generation overflow"))?;
            inner.metadata = Some(DenseWeightTensorMetadata {
                dtype: view.dtype,
                format,
                shape: Arc::from(view.shape),
                strides: Arc::from(view.strides),
                byte_len: view.byte_size(),
                source,
            });
            inner.keys.clear();
        }

        let sub_key = DenseWeightSubKey {
            role,
            expert,
            in_features,
            out_features,
        };
        if let Some(key) = inner.keys.get(&sub_key) {
            return Ok(ResolvedDenseWeight {
                key: Arc::clone(key),
                payload: None,
            });
        }

        let payload = payload()?;
        // The hash is computed once for diagnostics/collision hardening, but it
        // is never the sole identity: the immutable kernel slot generation,
        // source provenance, role/expert, dtype, format, dimensions, and layout
        // all participate in equality.
        let content_hash = stable_bytes_hash(&payload);
        #[cfg(test)]
        {
            inner.hashed_bytes = inner.hashed_bytes.saturating_add(payload.len());
            if matches!(payload, Cow::Owned(_)) {
                inner.materialized_bytes = inner.materialized_bytes.saturating_add(payload.len());
            }
        }
        let metadata = inner
            .metadata
            .as_ref()
            .expect("constant packed weight metadata was just initialized");
        let key = Arc::new(DenseWeightCacheKey {
            identity_generation: inner.generation,
            dtype: metadata.dtype,
            format,
            in_features,
            out_features,
            role,
            expert,
            tensor_shape: Arc::clone(&metadata.shape),
            tensor_strides: Arc::clone(&metadata.strides),
            byte_len: payload.len(),
            content_hash,
            source,
        });
        inner.keys.insert(sub_key, Arc::clone(&key));
        Ok(ResolvedDenseWeight {
            key,
            payload: Some(payload),
        })
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> (usize, usize, usize) {
        let inner = self.inner.lock().unwrap();
        (
            inner.keys.len(),
            inner.hashed_bytes,
            inner.materialized_bytes,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(super) struct DenseWeightCacheKey {
    identity_generation: u64,
    dtype: DataType,
    format: BlockFormat,
    in_features: usize,
    out_features: usize,
    role: u8,
    expert: Option<usize>,
    tensor_shape: Arc<[usize]>,
    tensor_strides: Arc<[i64]>,
    byte_len: usize,
    content_hash: u64,
    source: DenseWeightSource,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum DenseWeightSource {
    ExternalMmap {
        mapping_id: usize,
        offset: usize,
        mapped_len: usize,
        len: usize,
    },
    OpaqueConstant {
        base_ptr: usize,
        byte_offset: usize,
        len: usize,
    },
}

impl DenseWeightSource {
    fn from_tensor(view: &TensorView) -> Result<Self> {
        Ok(match view.backing {
            TensorBacking::ExternalMmap(region) => Self::ExternalMmap {
                mapping_id: region.mapping_id,
                offset: region
                    .offset
                    .checked_add(view.byte_offset)
                    .ok_or_else(|| error("external packed weight byte offset overflow"))?,
                mapped_len: region.len,
                len: view.byte_size(),
            },
            TensorBacking::Opaque => Self::OpaqueConstant {
                base_ptr: view.data.0 as usize,
                byte_offset: view.byte_offset,
                len: view.byte_size(),
            },
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum DenseWeightCacheStatus {
    Hit,
    MissStored,
    MissNotStored,
}

#[derive(Default)]
pub(super) struct DenseWeightCache {
    max_bytes: Option<usize>,
    inner: Mutex<DenseWeightCacheInner>,
}

#[derive(Default)]
struct DenseWeightCacheInner {
    used_bytes: usize,
    tick: u64,
    entries: HashMap<DenseWeightCacheKey, DenseWeightCacheEntry>,
    #[cfg(test)]
    hits: usize,
    #[cfg(test)]
    builds: usize,
}

struct DenseWeightCacheEntry {
    value: Arc<Vec<f32>>,
    bytes: usize,
    last_used: u64,
}

impl DenseWeightCache {
    pub(super) fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    pub(super) fn with_limit(limit_bytes: usize) -> Self {
        Self {
            max_bytes: Some(limit_bytes),
            inner: Mutex::new(DenseWeightCacheInner {
                used_bytes: 0,
                tick: 0,
                entries: HashMap::new(),
                hits: 0,
                builds: 0,
            }),
        }
    }

    pub(super) fn get_or_insert_with(
        &self,
        key: &DenseWeightCacheKey,
        build: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<(Arc<Vec<f32>>, DenseWeightCacheStatus)> {
        let max_bytes = self.max_bytes.unwrap_or_else(dense_weight_cache_bytes);
        self.get_or_insert_with_limit(key, max_bytes, build)
    }

    fn get_or_insert_with_limit(
        &self,
        key: &DenseWeightCacheKey,
        max_bytes: usize,
        build: impl FnOnce() -> Result<Vec<f32>>,
    ) -> Result<(Arc<Vec<f32>>, DenseWeightCacheStatus)> {
        let mut inner = self
            .inner
            .lock()
            .expect("BlockQuantized dense cache lock poisoned");
        let next_tick = inner.tick.wrapping_add(1);
        inner.tick = next_tick;
        if let Some(entry) = inner.entries.get_mut(key) {
            entry.last_used = next_tick;
            let value = Arc::clone(&entry.value);
            #[cfg(test)]
            {
                inner.hits = inner.hits.saturating_add(1);
            }
            return Ok((value, DenseWeightCacheStatus::Hit));
        }

        // Deliberately build under the per-kernel mutex. Dense expansion is a
        // rare cold-path operation, while serializing it guarantees single
        // flight: concurrent misses cannot duplicate a large allocation and
        // temporarily violate the configured resident-byte bound. Builders are
        // pure dequantizers and never re-enter this cache.
        let value = Arc::new(build()?);
        #[cfg(test)]
        {
            inner.builds = inner.builds.saturating_add(1);
        }
        let bytes = value
            .len()
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| error("cached dense weight byte count overflow"))?;
        if max_bytes == 0 || bytes > max_bytes {
            return Ok((value, DenseWeightCacheStatus::MissNotStored));
        }
        while inner.used_bytes > max_bytes.saturating_sub(bytes) {
            let Some(oldest) = inner
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| key.clone())
            else {
                break;
            };
            let removed = inner
                .entries
                .remove(&oldest)
                .expect("oldest dense cache entry must still exist");
            inner.used_bytes = inner
                .used_bytes
                .checked_sub(removed.bytes)
                .expect("dense cache byte accounting underflow");
        }
        inner.used_bytes = inner
            .used_bytes
            .checked_add(bytes)
            .ok_or_else(|| error("cached dense weight aggregate byte count overflow"))?;
        inner.entries.insert(
            key.clone(),
            DenseWeightCacheEntry {
                value: Arc::clone(&value),
                bytes,
                last_used: next_tick,
            },
        );
        Ok((value, DenseWeightCacheStatus::MissStored))
    }

    #[cfg(test)]
    pub(super) fn stats(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.entries.len(), inner.used_bytes)
    }

    #[cfg(test)]
    pub(super) fn activity(&self) -> (usize, usize) {
        let inner = self.inner.lock().unwrap();
        (inner.hits, inner.builds)
    }
}

fn dense_weight_cache_bytes() -> usize {
    static BYTES: OnceLock<usize> = OnceLock::new();
    *BYTES.get_or_init(|| {
        parse_dense_weight_cache_bytes(std::env::var(DENSE_WEIGHT_CACHE_BYTES_ENV).ok().as_deref())
    })
}

fn parse_dense_weight_cache_bytes(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_DENSE_WEIGHT_CACHE_BYTES)
}

pub(super) fn packed_tensor_bytes<'a>(view: &'a TensorView<'_>) -> Result<Cow<'a, [u8]>> {
    view.validate()?;
    if view.dtype != DataType::Uint8 {
        return Err(error(format!(
            "packed weight dtype {:?} unsupported; expected Uint8",
            view.dtype
        )));
    }
    if view.is_contiguous() {
        let len = view.byte_size();
        if len == 0 {
            return Ok(Cow::Borrowed(&[]));
        }
        // SAFETY: `view` is validated, contiguous Uint8 storage, and the
        // executor has bounds-checked the logical byte extent against the live
        // backing allocation. The borrow cannot outlive the input view.
        return Ok(Cow::Borrowed(unsafe {
            std::slice::from_raw_parts(view.data_ptr::<u8>(), len)
        }));
    }
    Ok(Cow::Owned(to_dense_bytes(view)?))
}

fn stable_bytes_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

pub(super) fn dequantize_weight_kn(
    format: BlockFormat,
    k: usize,
    n: usize,
    packed: &[u8],
) -> Result<Vec<f32>> {
    let qk = format.qk();
    let block_bytes = format.block_bytes();
    let blocks = k.div_ceil(qk);
    let expected_bytes = n
        .checked_mul(blocks)
        .and_then(|value| value.checked_mul(block_bytes))
        .ok_or_else(|| error("packed_B byte count overflow"))?;
    if packed.len() != expected_bytes {
        return Err(error(format!(
            "packed_B must contain exactly {expected_bytes} bytes, got {}",
            packed.len()
        )));
    }

    let weight_elements = k
        .checked_mul(n)
        .ok_or_else(|| error("dequantized weight element count overflow"))?;
    let mut weight_kn = vec![0.0f32; weight_elements];
    let block_row_elements = qk
        .min(k)
        .checked_mul(n)
        .ok_or_else(|| error("dequantized block-row element count overflow"))?;
    let decoder = format.decoder();
    weight_kn
        .par_chunks_mut(block_row_elements)
        .enumerate()
        .for_each(|(block_index, weight_rows)| {
            let mut decoded = [0.0f32; IQ_SUPER_QK];
            let valid = weight_rows.len() / n;
            for output in 0..n {
                let packed_start = (output * blocks + block_index) * block_bytes;
                decoder(
                    &packed[packed_start..packed_start + block_bytes],
                    &mut decoded[..qk],
                );
                for (offset, value) in decoded[..valid].iter().copied().enumerate() {
                    weight_rows[offset * n + output] = value;
                }
            }
        });
    Ok(weight_kn)
}

fn decode_mxfp4_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), MXFP4_BLOCK_BYTES);
    debug_assert_eq!(output.len(), MXFP4_QK);
    let scale = decode_e8m0_scale(block[0]);
    for j in 0..16 {
        let packed = block[1 + j];
        output[j] = decode_mxfp4_value(packed, scale);
        output[j + 16] = decode_mxfp4_value(packed >> 4, scale);
    }
}

fn decode_mxfp4_value(code: u8, scale: f32) -> f32 {
    let value = decode_e2m1(code);
    if value == 0.0 { 0.0 } else { value * scale }
}

#[cfg(any(target_arch = "x86_64", test))]
fn e8m0_half_scale(exponent: u8) -> f32 {
    decode_e8m0_scale(exponent) * 0.5
}

fn decode_iq4_nl_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_NL_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ4_NL_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for j in 0..16 {
        let packed = block[2 + j];
        output[j] = scale * IQ4_NL_CODEBOOK[(packed & 0x0f) as usize] as f32;
        output[j + 16] = scale * IQ4_NL_CODEBOOK[(packed >> 4) as usize] as f32;
    }
}

fn decode_iq4_xs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_XS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let quants = &block[8..];
    for subblock in 0..8 {
        let low = (scales_l[subblock / 2] >> (4 * (subblock % 2))) & 0x0f;
        let high = ((scales_h >> (2 * subblock)) & 0x03) as u8;
        let subscale = scale * f32::from((low | (high << 4)) as i8 - 32);
        let output = &mut output[subblock * 32..][..32];
        let quants = &quants[subblock * 16..][..16];
        for j in 0..16 {
            output[j] = subscale * IQ4_NL_CODEBOOK[(quants[j] & 0x0f) as usize] as f32;
            output[j + 16] = subscale * IQ4_NL_CODEBOOK[(quants[j] >> 4) as usize] as f32;
        }
    }
}

#[cfg(target_arch = "x86_64")]
fn decode_mxfp4_block_avx2_dispatch(block: &[u8], output: &mut [f32]) {
    // SAFETY: BlockFormat::decoder selects this wrapper only after AVX2 detection.
    unsafe { decode_mxfp4_block_avx2(block, output) }
}

#[cfg(target_arch = "x86_64")]
fn decode_iq4_nl_block_avx2_dispatch(block: &[u8], output: &mut [f32]) {
    // SAFETY: BlockFormat::decoder selects this wrapper only after AVX2 detection.
    unsafe { decode_iq4_nl_block_avx2(block, output) }
}

#[cfg(target_arch = "x86_64")]
fn decode_iq4_xs_block_avx2_dispatch(block: &[u8], output: &mut [f32]) {
    // SAFETY: BlockFormat::decoder selects this wrapper only after AVX2 detection.
    unsafe { decode_iq4_xs_block_avx2(block, output) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_mxfp4_block_avx2(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), MXFP4_BLOCK_BYTES);
    debug_assert_eq!(output.len(), MXFP4_QK);
    let half_scale = e8m0_half_scale(block[0]);
    // SAFETY: the block and output lengths above cover the 16-byte load and 32 outputs.
    unsafe {
        decode_nibbles_scaled_avx2(&block[1..], &E2M1_DOUBLED, half_scale, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_iq4_nl_block_avx2(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_NL_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ4_NL_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    // SAFETY: the block and output lengths above cover the 16-byte load and 32 outputs.
    unsafe {
        decode_nibbles_scaled_avx2(&block[2..], &IQ4_NL_CODEBOOK, scale, output);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_iq4_xs_block_avx2(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ4_XS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let scales_h = u16::from_le_bytes([block[2], block[3]]);
    let scales_l = &block[4..8];
    let quants = &block[8..];
    for subblock in 0..8 {
        let low = (scales_l[subblock / 2] >> (4 * (subblock % 2))) & 0x0f;
        let high = ((scales_h >> (2 * subblock)) & 0x03) as u8;
        let subscale = scale * f32::from((low | (high << 4)) as i8 - 32);
        // SAFETY: each subblock owns 16 packed bytes and 32 decoded outputs.
        unsafe {
            decode_nibbles_scaled_avx2(
                &quants[subblock * 16..][..16],
                &IQ4_NL_CODEBOOK,
                subscale,
                &mut output[subblock * 32..][..32],
            );
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn decode_nibbles_scaled_avx2(
    packed: &[u8],
    codebook: &[i8; 16],
    scale: f32,
    output: &mut [f32],
) {
    use std::arch::x86_64::*;

    debug_assert!(packed.len() >= 16);
    debug_assert!(output.len() >= 32);
    // SAFETY: callers provide at least 16 packed bytes and 32 output elements.
    let (low_values, high_values) = unsafe {
        let bytes = _mm_loadu_si128(packed.as_ptr().cast());
        let mask = _mm_set1_epi8(0x0f);
        let low_indices = _mm_and_si128(bytes, mask);
        let high_indices = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
        let table = _mm_loadu_si128(codebook.as_ptr().cast());
        (
            _mm_shuffle_epi8(table, low_indices),
            _mm_shuffle_epi8(table, high_indices),
        )
    };
    let scale = _mm256_set1_ps(scale);
    // SAFETY: each store writes eight elements inside the validated output slice.
    unsafe {
        let low0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low_values));
        let low1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(low_values)));
        let high0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high_values));
        let high1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128::<8>(high_values)));
        _mm256_storeu_ps(output.as_mut_ptr(), _mm256_mul_ps(low0, scale));
        _mm256_storeu_ps(output.as_mut_ptr().add(8), _mm256_mul_ps(low1, scale));
        _mm256_storeu_ps(output.as_mut_ptr().add(16), _mm256_mul_ps(high0, scale));
        _mm256_storeu_ps(output.as_mut_ptr().add(24), _mm256_mul_ps(high1, scale));
    }
}

fn decode_iq3_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ3_S_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..66];
    let high_bits = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];

    for group64 in 0..4 {
        let packed_scale = scales[group64];
        for half in 0..2 {
            let subscale = scale * f32::from(1 + 2 * ((packed_scale >> (4 * half)) & 0x0f));
            let qh = high_bits[group64 * 2 + half];
            let quant_base = group64 * 16 + half * 8;
            let sign_base = group64 * 8 + half * 4;
            let output_base = group64 * 64 + half * 32;
            for vector in 0..4 {
                let index0 = usize::from(quants[quant_base + 2 * vector])
                    | (usize::from((qh >> (2 * vector)) & 1) << 8);
                let index1 = usize::from(quants[quant_base + 2 * vector + 1])
                    | (usize::from((qh >> (2 * vector + 1)) & 1) << 8);
                let grid0 = IQ3S_GRID[index0];
                let grid1 = IQ3S_GRID[index1];
                let sign_mask = signs[sign_base + vector];
                let vector_base = output_base + vector * 8;
                for j in 0..4 {
                    let magnitude0 = ((grid0 >> (8 * j)) & 0xff) as f32;
                    let magnitude1 = ((grid1 >> (8 * j)) & 0xff) as f32;
                    output[vector_base + j] = if sign_mask & (1 << j) != 0 {
                        -subscale * magnitude0
                    } else {
                        subscale * magnitude0
                    };
                    output[vector_base + j + 4] = if sign_mask & (1 << (j + 4)) != 0 {
                        -subscale * magnitude1
                    } else {
                        subscale * magnitude1
                    };
                }
            }
        }
    }
}

fn decode_iq3_xxs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ3_XXS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..66];
    let scales_and_signs = &block[66..98];

    for group32 in 0..8 {
        let metadata_base = group32 * 4;
        let metadata = u32::from_le_bytes([
            scales_and_signs[metadata_base],
            scales_and_signs[metadata_base + 1],
            scales_and_signs[metadata_base + 2],
            scales_and_signs[metadata_base + 3],
        ]);
        let subscale = scale * (0.5 + (metadata >> 28) as f32) * 0.5;
        let quant_base = group32 * 8;
        for vector in 0..4 {
            let sign_mask = IQ2XS_SIGNS[((metadata >> (7 * vector)) & 127) as usize];
            let grid0 = IQ3XXS_GRID[quants[quant_base + 2 * vector] as usize];
            let grid1 = IQ3XXS_GRID[quants[quant_base + 2 * vector + 1] as usize];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..4 {
                let magnitude0 = ((grid0 >> (8 * j)) & 0xff) as f32;
                let magnitude1 = ((grid1 >> (8 * j)) & 0xff) as f32;
                output[output_base + j] = if sign_mask & (1 << j) != 0 {
                    -subscale * magnitude0
                } else {
                    subscale * magnitude0
                };
                output[output_base + j + 4] = if sign_mask & (1 << (j + 4)) != 0 {
                    -subscale * magnitude1
                } else {
                    subscale * magnitude1
                };
            }
        }
    }
}

fn decode_iq2_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_S_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..34];
    let signs = &block[34..66];
    let high_bits = &block[66..74];
    let scales = &block[74..82];

    for (group32, &packed_scale) in scales.iter().enumerate() {
        let qh = high_bits[group32];
        for vector in 0..4 {
            let subscale =
                scale * (0.5 + ((packed_scale >> (4 * (vector / 2))) & 0x0f) as f32) * 0.25;
            let index = usize::from(quants[group32 * 4 + vector])
                | (usize::from((qh >> (2 * vector)) & 0x03) << 8);
            let grid = IQ2S_GRID[index];
            let sign_mask = signs[group32 * 4 + vector];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let magnitude = ((grid >> (8 * j)) & 0xff) as f32;
                output[output_base + j] = if sign_mask & (1 << j) != 0 {
                    -subscale * magnitude
                } else {
                    subscale * magnitude
                };
            }
        }
    }
}

fn decode_iq2_xs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_XS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..66];
    let scales = &block[66..74];

    for (group32, &packed_scale) in scales.iter().enumerate() {
        for vector in 0..4 {
            let quant_base = group32 * 8 + vector * 2;
            let quant = u16::from_le_bytes([quants[quant_base], quants[quant_base + 1]]);
            let subscale =
                scale * (0.5 + ((packed_scale >> (4 * (vector / 2))) & 0x0f) as f32) * 0.25;
            let grid = IQ2XS_GRID[usize::from(quant & 511)];
            let sign_mask = IQ2XS_SIGNS[usize::from(quant >> 9)];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let magnitude = ((grid >> (8 * j)) & 0xff) as f32;
                output[output_base + j] = if sign_mask & (1 << j) != 0 {
                    -subscale * magnitude
                } else {
                    subscale * magnitude
                };
            }
        }
    }
}

fn decode_iq2_xxs_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ2_XXS_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    for group32 in 0..8 {
        let base = 2 + group32 * 8;
        let metadata = u32::from_le_bytes([
            block[base + 4],
            block[base + 5],
            block[base + 6],
            block[base + 7],
        ]);
        let subscale = scale * (0.5 + (metadata >> 28) as f32) * 0.25;
        for vector in 0..4 {
            let grid = IQ2XXS_GRID[block[base + vector] as usize];
            let sign_mask = IQ2XS_SIGNS[((metadata >> (7 * vector)) & 127) as usize];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let magnitude = ((grid >> (8 * j)) & 0xff) as f32;
                output[output_base + j] = if sign_mask & (1 << j) != 0 {
                    -subscale * magnitude
                } else {
                    subscale * magnitude
                };
            }
        }
    }
}

fn decode_iq1_s_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ1_S_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let quants = &block[2..34];
    let high_bits = &block[34..50];

    for group32 in 0..8 {
        let high_base = group32 * 2;
        let qh = u16::from_le_bytes([high_bits[high_base], high_bits[high_base + 1]]);
        let subscale = scale * f32::from(2 * ((qh >> 12) & 7) + 1);
        let delta = if qh & 0x8000 != 0 {
            -IQ1_S_DELTA
        } else {
            IQ1_S_DELTA
        };
        for vector in 0..4 {
            let index = usize::from(quants[group32 * 4 + vector])
                | (usize::from((qh >> (3 * vector)) & 7) << 8);
            let grid = IQ1S_GRID[index];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let value = ((grid >> (8 * j)) & 0xff) as u8 as i8;
                output[output_base + j] = subscale * (f32::from(value) + delta);
            }
        }
    }
}

fn decode_iq1_m_block(block: &[u8], output: &mut [f32]) {
    debug_assert_eq!(block.len(), IQ1_M_BLOCK_BYTES);
    debug_assert_eq!(output.len(), IQ_SUPER_QK);
    let quants = &block[..32];
    let high_bits = &block[32..48];
    let scales = &block[48..56];
    let packed_scales = [
        u16::from_le_bytes([scales[0], scales[1]]),
        u16::from_le_bytes([scales[2], scales[3]]),
        u16::from_le_bytes([scales[4], scales[5]]),
        u16::from_le_bytes([scales[6], scales[7]]),
    ];
    let scale_bits = (packed_scales[0] >> 12)
        | ((packed_scales[1] >> 8) & 0x00f0)
        | ((packed_scales[2] >> 4) & 0x0f00)
        | (packed_scales[3] & 0xf000);
    let scale = half::f16::from_bits(scale_bits).to_f32();

    for group32 in 0..8 {
        let packed_scale = packed_scales[group32 / 2];
        let scale_shift = 6 * (group32 % 2);
        let subscale1 = scale * f32::from(2 * ((packed_scale >> scale_shift) & 7) + 1);
        let subscale2 = scale * f32::from(2 * ((packed_scale >> (scale_shift + 3)) & 7) + 1);
        for vector in 0..4 {
            let qh = high_bits[group32 * 2 + vector / 2];
            let high_shift = 4 * (vector % 2);
            let index = usize::from(quants[group32 * 4 + vector])
                | (usize::from((qh >> high_shift) & 7) << 8);
            let delta = if qh & (0x08 << high_shift) != 0 {
                -IQ1_M_DELTA
            } else {
                IQ1_M_DELTA
            };
            let subscale = if vector < 2 { subscale1 } else { subscale2 };
            let grid = IQ1S_GRID[index];
            let output_base = group32 * 32 + vector * 8;
            for j in 0..8 {
                let value = ((grid >> (8 * j)) & 0xff) as u8 as i8;
                output[output_base + j] = subscale * (f32::from(value) + delta);
            }
        }
    }
}

fn required_positive_attr(node: &Node, name: &str) -> Result<usize> {
    let value = optional_int_attr(node, name)?
        .ok_or_else(|| error(format!("missing required integer attribute '{name}'")))?;
    if value <= 0 {
        return Err(error(format!(
            "attribute '{name}' must be positive, got {value}"
        )));
    }
    Ok(value as usize)
}

fn optional_int_attr(node: &Node, name: &str) -> Result<Option<i64>> {
    match node.attr(name) {
        Some(attribute) => attribute
            .as_int()
            .map(Some)
            .ok_or_else(|| error(format!("attribute '{name}' must be an integer"))),
        None => Ok(None),
    }
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn require_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{DOMAIN}::{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    use onnx_runtime_loader::Model;

    fn model_node(
        format: &str,
        a_shape: &[usize],
        b_shape: &[usize],
        output_shape: &[usize],
        k: usize,
        n: usize,
        with_bias: bool,
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(DOMAIN.into(), 1);
        let a = graph.create_named_value(
            "A",
            DataType::Float32,
            static_shape(a_shape.iter().copied()),
        );
        graph.add_input(a);
        let packed_b = graph.create_named_value(
            "packed_B",
            DataType::Uint8,
            static_shape(b_shape.iter().copied()),
        );
        graph.add_input(packed_b);
        let mut inputs = vec![Some(a), Some(packed_b)];
        if with_bias {
            let bias = graph.create_named_value("bias", DataType::Float32, static_shape([n]));
            graph.add_input(bias);
            inputs.push(Some(bias));
        }
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), OP, inputs, vec![output]);
        node.domain = DOMAIN.into();
        node.attributes.insert("K".into(), Attribute::Int(k as i64));
        node.attributes.insert("N".into(), Attribute::Int(n as i64));
        node.attributes.insert(
            "format".into(),
            Attribute::String(format.as_bytes().to_vec()),
        );
        node.attributes
            .insert("block_layout_version".into(), Attribute::Int(1));
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn kernel(graph: &Graph, node: NodeId) -> Box<dyn Kernel> {
        let model = Model::new(graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .expect("CPU EP must register BlockQuantizedMatMul")
    }

    #[test]
    fn mxfp4_known_block_matches_ocp_e2m1_and_llama_layout() {
        let mut packed = vec![127u8];
        packed.extend((0u8..16).map(|code| code | (code << 4)));
        let view = Owned::u8(&[1, 1, 17], &packed);
        let kernel = BlockQuantizedMatMulKernel {
            k: 32,
            n: 1,
            format: BlockFormat::Mxfp4,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = kernel.dequantize_weight_kn(&view.view()).unwrap();
        let values = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let mut expected = Vec::with_capacity(32);
        expected.extend(values);
        expected.extend(values.map(|value| -value));
        expected.extend(values);
        expected.extend(values.map(|value| -value));
        assert_eq!(actual, expected);
    }

    #[test]
    fn e8m0_decode_covers_subnormal_extremes_and_nan() {
        assert_eq!((e8m0_half_scale(0) * 2.0).to_bits(), 0x0040_0000);
        assert_eq!((e8m0_half_scale(1) * 2.0).to_bits(), 0x0080_0000);
        assert_eq!(e8m0_half_scale(127), 0.5);
        assert_eq!(e8m0_half_scale(128), 1.0);
        assert_eq!((e8m0_half_scale(254) * 2.0).to_bits(), 0x7f00_0000);
        assert!(e8m0_half_scale(255).is_nan());
    }

    #[test]
    fn mxfp4_batched_matmul_with_partial_block_and_bias_matches_reference() {
        let (m, k, n): (usize, usize, usize) = (2, 45, 2);
        let blocks = k.div_ceil(32);
        let mut packed = vec![0u8; n * blocks * MXFP4_BLOCK_BYTES];
        let mut weight_nk = vec![0.0f32; n * k];
        for output in 0..n {
            for block in 0..blocks {
                let start = (output * blocks + block) * MXFP4_BLOCK_BYTES;
                packed[start] = 127 + output as u8;
                for j in 0..16 {
                    let low = ((j + block + output) % 16) as u8;
                    let high = ((15 + output - (j % 2)) % 16) as u8;
                    packed[start + 1 + j] = low | (high << 4);
                }
                let mut decoded = [0.0; 32];
                decode_mxfp4_block(&packed[start..start + MXFP4_BLOCK_BYTES], &mut decoded);
                for offset in 0..(k - block * 32).min(32) {
                    weight_nk[output * k + block * 32 + offset] = decoded[offset];
                }
            }
        }
        let activations: Vec<f32> = (0..m * k)
            .map(|index| ((index * 7 % 19) as f32 - 9.0) / 8.0)
            .collect();
        let bias = [0.25, -0.5];
        let mut expected = vec![0.0; m * n];
        for row in 0..m {
            for output in 0..n {
                expected[row * n + output] = bias[output]
                    + (0..k)
                        .map(|inner| activations[row * k + inner] * weight_nk[output * k + inner])
                        .sum::<f32>();
            }
        }

        let (graph, node) = model_node("mxfp4", &[m, k], &[n, blocks, 17], &[m, n], k, n, true);
        let kernel = kernel(&graph, node);
        let a = Owned::f32(&[m, k], &activations);
        let b = Owned::u8(&[n, blocks, 17], &packed);
        let bias = Owned::f32(&[n], &bias);
        let mut y = Owned::zeros_f32(&[m, n]);
        kernel
            .execute(&[a.view(), b.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();
        for (actual, expected) in y.to_f32().iter().zip(expected) {
            assert!((actual - expected).abs() <= 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn block_quantized_matmul_manifest_counter_proves_dense_f32_dequant_dispatch() {
        let before =
            BLOCK_QUANTIZED_MATMUL_DENSE_F32_TEST_HITS.load(std::sync::atomic::Ordering::Relaxed);
        let (graph, node) = model_node("mxfp4", &[1, 32], &[1, 1, 17], &[1, 1], 32, 1, false);
        let kernel = kernel(&graph, node);
        let mut packed = vec![127u8];
        packed.extend([0u8; 16]);
        let a = Owned::f32(&[1, 32], &[1.0; 32]);
        let b = Owned::u8(&[1, 1, 17], &packed);
        let mut y = Owned::zeros_f32(&[1, 1]);

        kernel
            .execute(&[a.view(), b.view()], &mut [y.view_mut()])
            .unwrap();

        let after =
            BLOCK_QUANTIZED_MATMUL_DENSE_F32_TEST_HITS.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            after > before,
            "dispatch manifest counter must prove BlockQuantizedMatMul ran the dense-f32 dequant path"
        );
        assert_eq!(y.to_f32(), vec![0.0]);
    }

    #[test]
    fn mxfp4_bfloat16_decode_and_prefill_match_widened_reference() {
        let (k, n) = (32, 2);
        let mut packed = vec![0u8; n * MXFP4_BLOCK_BYTES];
        for output in 0..n {
            let start = output * MXFP4_BLOCK_BYTES;
            packed[start] = 127 + output as u8;
            for index in 0..16 {
                let low = ((index + output) % 16) as u8;
                let high = ((15 + output - index % 2) % 16) as u8;
                packed[start + 1 + index] = low | (high << 4);
            }
        }
        let packed_weight = Owned::u8(&[n, 1, MXFP4_BLOCK_BYTES], &packed);
        let kernel = BlockQuantizedMatMulKernel {
            k,
            n,
            format: BlockFormat::Mxfp4,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let weight = kernel.dequantize_weight_kn(&packed_weight.view()).unwrap();

        for rows in [1, 4] {
            let activation_values: Vec<f32> = (0..rows * k)
                .map(|index| ((index * 13 % 31) as f32 - 15.0) * 0.0625)
                .collect();
            let activation = Owned::bf16(&[rows, k], &activation_values);
            let bias = Owned::bf16(&[n], &[0.375, -0.625]);
            let mut output = Owned::zeros(DataType::BFloat16, &[rows, n]);
            kernel
                .execute(
                    &[activation.view(), packed_weight.view(), bias.view()],
                    &mut [output.view_mut()],
                )
                .unwrap();

            let widened_activation = activation.to_bf16_as_f32();
            let widened_bias = bias.to_bf16_as_f32();
            let mut expected = vec![0.0; rows * n];
            for row in 0..rows {
                for column in 0..n {
                    expected[row * n + column] = widened_bias[column]
                        + (0..k)
                            .map(|inner| {
                                widened_activation[row * k + inner] * weight[inner * n + column]
                            })
                            .sum::<f32>();
                }
            }
            for (index, (actual, expected)) in output
                .to_bf16_as_f32()
                .into_iter()
                .zip(expected)
                .enumerate()
            {
                let tolerance = 5e-3 + 1e-2 * expected.abs();
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "rows {rows}, element {index}: {actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn iq4_nl_uses_llama_codebook_and_fp16_scale() {
        let mut packed = half::f16::from_f32(0.5).to_le_bytes().to_vec();
        packed.extend((0u8..16).map(|code| code | ((15 - code) << 4)));
        let view = Owned::u8(&[1, 1, IQ4_NL_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: 32,
            n: 1,
            format: BlockFormat::Iq4Nl,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let expected: Vec<f32> = IQ4_NL_CODEBOOK
            .iter()
            .chain(IQ4_NL_CODEBOOK.iter().rev())
            .map(|value| *value as f32 * 0.5)
            .collect();
        assert_eq!(actual, expected);

        let activation: Vec<f32> = (1..=32).map(|value| value as f32 / 16.0).collect();
        let reference = activation
            .iter()
            .zip(&expected)
            .map(|(a, b)| a * b)
            .sum::<f32>();
        let (graph, node) = model_node("iq4_nl", &[1, 32], &[1, 1, 18], &[1, 1], 32, 1, false);
        let kernel = kernel(&graph, node);
        let a = Owned::f32(&[1, 32], &activation);
        let b = Owned::u8(&[1, 1, 18], &packed);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel
            .execute(&[a.view(), b.view()], &mut [y.view_mut()])
            .unwrap();
        assert!((y.to_f32()[0] - reference).abs() <= 1e-5);
    }

    #[test]
    fn iq4_xs_decodes_six_bit_subscales_and_iq4_nl_values() {
        let mut packed = half::f16::from_f32(0.5).to_le_bytes().to_vec();
        packed.extend(0xaaaau16.to_le_bytes());
        packed.extend([0x55; 4]);
        packed.extend([0x98; 128]);
        let view = Owned::u8(&[1, 1, IQ4_XS_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq4Xs,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let mut expected = Vec::with_capacity(IQ_SUPER_QK);
        for _ in 0..8 {
            expected.extend([2.5; 16]);
            expected.extend([32.5; 16]);
        }
        // First sub-block: ls = 0b10_0101 = 37, dl = 0.5*(37-32) = 2.5.
        // Byte 0x98 therefore gives 2.5*codebook[8] = 2.5 at y[0] and
        // 2.5*codebook[9] = 32.5 at y[16].
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq3_s_decodes_grid_high_bits_signs_and_odd_subscales() {
        let mut packed = vec![0u8; IQ3_S_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes());
        packed[66] = 0x01;
        packed[74] = 0x81;
        packed[106..110].fill(0x10);
        let view = Owned::u8(&[1, 1, IQ3_S_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq3S,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let mut expected = vec![0.0; IQ_SUPER_QK];
        for group64 in 0..4 {
            expected[group64 * 64..group64 * 64 + 32].fill(0.5);
            expected[group64 * 64 + 32..group64 * 64 + 64].fill(1.5);
        }
        expected[0..4].copy_from_slice(&[-3.5, 2.5, 4.5, 2.5]);
        expected[7] = -0.5;
        // qh bit zero raises the first index to 256, whose grid is {7,5,9,5};
        // the paired zero index is {1,1,1,1}. Scale byte 0x10 gives db1=0.5
        // and db2=1.5. signs[0]=0x81 negates weights zero and seven.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq3_xxs_decodes_two_grids_packed_signs_and_scale() {
        let mut packed = vec![0u8; IQ3_XXS_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        let metadata = 2u32 << 28 | 3u32 << 21 | 2u32 << 14 | 1u32 << 7;
        for group32 in 0..8 {
            packed[2 + group32 * 8..2 + group32 * 8 + 8]
                .copy_from_slice(&[0, 255, 1, 254, 2, 253, 3, 252]);
            packed[66 + group32 * 4..66 + group32 * 4 + 4].copy_from_slice(&metadata.to_le_bytes());
        }
        let view = Owned::u8(&[1, 1, IQ3_XXS_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq3Xxs,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let group = [
            10.0, 10.0, 10.0, 10.0, 10.0, 70.0, 130.0, 155.0, -50.0, 10.0, 10.0, 10.0, 90.0, 50.0,
            110.0, -155.0, 90.0, -10.0, 10.0, 10.0, 50.0, 10.0, 110.0, -155.0, -30.0, -30.0, 10.0,
            10.0, 10.0, 10.0, 110.0, 155.0,
        ];
        let expected: Vec<f32> = group.into_iter().cycle().take(IQ_SUPER_QK).collect();
        // scale4=2 gives db=2*(0.5+2)*0.5=2.5. The first pair uses
        // grids 0={4,4,4,4} and 255={4,28,52,62}; sign index zero is positive.
        // Sign indices 1,2,3 then apply masks 0x81, 0x82, and 0x03.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq2_s_decodes_ten_bit_grids_explicit_signs_and_nibble_scales() {
        let mut packed = vec![0u8; IQ2_S_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        for group32 in 0..8 {
            packed[2 + group32 * 4..2 + group32 * 4 + 4].copy_from_slice(&[0, 0, 0, 255]);
            packed[34 + group32 * 4..34 + group32 * 4 + 4]
                .copy_from_slice(&[0x00, 0x81, 0x82, 0x03]);
            packed[66 + group32] = 0xe4;
            packed[74 + group32] = 0x21;
        }
        let view = Owned::u8(&[1, 1, IQ2_S_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq2S,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let group = [
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, -18.75, 18.75, 18.75, 18.75, 18.75, 6.0, 18.75,
            -6.0, 31.25, -31.25, 53.75, 10.0, 31.25, 10.0, 10.0, -31.25, -53.75, -53.75, 53.75,
            53.75, 53.75, 53.75, 53.75, 53.75,
        ];
        let expected: Vec<f32> = group.into_iter().cycle().take(IQ_SUPER_QK).collect();
        // qh=0xe4 combines low indices {0,0,0,255} into {0,256,512,1023}.
        // Scale byte 0x21 gives db={0.75,1.25}; signs are explicit per vector.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq2_xs_decodes_nine_bit_grids_sign_table_and_nibble_scales() {
        let mut packed = vec![0u8; IQ2_XS_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        let grids = [0u16, 511, 1, 510];
        for group32 in 0..8 {
            for (vector, grid) in grids.into_iter().enumerate() {
                let quant = grid | ((vector as u16) << 9);
                let base = 2 + group32 * 8 + vector * 2;
                packed[base..base + 2].copy_from_slice(&quant.to_le_bytes());
            }
            packed[66 + group32] = 0x21;
        }
        let view = Owned::u8(&[1, 1, IQ2_XS_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq2Xs,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let group = [
            6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, -32.25, 32.25, 32.25, 32.25, 32.25, 32.25,
            32.25, -32.25, 53.75, -10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -10.0, -31.25, -10.0, 31.25,
            53.75, 53.75, 53.75, 53.75, 53.75,
        ];
        let expected: Vec<f32> = group.into_iter().cycle().take(IQ_SUPER_QK).collect();
        // Scale byte 0x21 gives db={0.75,1.25}. Grid 511 is all 43s,
        // while grid 510 is {25,8,25,43,43,43,43,43}; sign indices 0..3
        // select masks 0x00, 0x81, 0x82, and 0x03.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq2_xxs_decodes_packed_grid_sign_and_scale_metadata() {
        let mut packed = vec![0u8; IQ2_XXS_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        let metadata = 2u32 << 28 | 3u32 << 21 | 2u32 << 14 | 1u32 << 7;
        for group32 in 0..8 {
            let base = 2 + group32 * 8;
            packed[base..base + 4].copy_from_slice(&[0, 1, 254, 255]);
            packed[base + 4..base + 8].copy_from_slice(&metadata.to_le_bytes());
        }
        let view = Owned::u8(&[1, 1, IQ2_XXS_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq2Xxs,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let sign_masks = [0u8, 129, 130, 3];
        let grids = [
            [8, 8, 8, 8, 8, 8, 8, 8],
            [43, 8, 8, 8, 8, 8, 8, 8],
            [8, 8, 25, 25, 8, 43, 43, 43],
            [8, 25, 8, 8, 25, 43, 43, 43],
        ];
        let mut expected = Vec::with_capacity(IQ_SUPER_QK);
        for _ in 0..8 {
            for (sign_mask, grid) in sign_masks.into_iter().zip(grids) {
                for (j, magnitude) in grid.into_iter().enumerate() {
                    let value = magnitude as f32 * 1.25;
                    expected.push(if sign_mask & (1 << j) != 0 {
                        -value
                    } else {
                        value
                    });
                }
            }
        }
        // scale4=2 gives db=2*(0.5+2)*0.25=1.25. Grid indices 0,1,254,255
        // begin {8,...}, {43,8,...}, {8,8,25,...}, and {8,25,8,...}.
        // Sign indices 0,1,2,3 map to 0x00,0x81,0x82,0x03.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq1_s_decodes_eleven_bit_grids_odd_scale_and_delta() {
        let mut packed = vec![0u8; IQ1_S_BLOCK_BYTES];
        packed[..2].copy_from_slice(&half::f16::from_f32(2.0).to_le_bytes());
        packed[4] = 0xff;
        packed[34..36].copy_from_slice(&0xa1c0u16.to_le_bytes());
        let view = Owned::u8(&[1, 1, IQ1_S_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq1S,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let mut expected = vec![-1.75; IQ_SUPER_QK];
        expected[..32].fill(-11.25);
        expected[16..24].fill(8.75);
        // qh=0xa1c0 gives odd scale 5, negative delta, and index 2047
        // for vector two. With d=2, grid 0=-1 and grid 2047=+1:
        // 10*(-1-0.125)=-11.25 and 10*(1-0.125)=8.75.
        assert_eq!(actual, expected);
    }

    #[test]
    fn iq1_m_decodes_bitsliced_fp16_two_odd_scales_and_vector_deltas() {
        let mut packed = vec![0u8; IQ1_M_BLOCK_BYTES];
        packed[1] = 0xff;
        packed[2] = 0xff;
        packed[32] = 0xf0;
        packed[33] = 0x8f;
        packed[48..56].copy_from_slice(&[0x1a, 0, 0, 0, 0, 0, 0, 0x40]);
        let view = Owned::u8(&[1, 1, IQ1_M_BLOCK_BYTES], &packed);
        let decoder = BlockQuantizedMatMulKernel {
            k: IQ_SUPER_QK,
            n: 1,
            format: BlockFormat::Iq1M,
            packed_b_constant: false,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        let actual = decoder.dequantize_weight_kn(&view.view()).unwrap();
        let mut expected = vec![-1.75; IQ_SUPER_QK];
        expected[..8].fill(-8.75);
        expected[8..16].fill(8.75);
        expected[16..24].fill(12.25);
        expected[24..32].fill(-15.75);
        // Scale high nibbles reconstruct fp16 0x4000 (2.0). sc[0]=0x001a
        // gives odd multipliers 5 and 7. qh selects grids 0,2047,2047,0
        // with deltas +,-,-,-, producing -8.75,8.75,12.25,-15.75.
        assert_eq!(actual, expected);
    }

    #[test]
    fn vendored_iq_grid_fingerprints_match_llama_cpp() {
        fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
            bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |hash, byte| {
                (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3)
            })
        }

        assert_eq!(fnv1a(IQ2XS_SIGNS), 0xf19b_a8f8_c329_2ba5);
        assert_eq!(
            fnv1a(IQ2XXS_GRID.into_iter().flat_map(u64::to_le_bytes)),
            0xbb4e_e025_b5ac_6e8e
        );
        assert_eq!(
            fnv1a(IQ3S_GRID.into_iter().flat_map(u32::to_le_bytes)),
            0xfa37_020c_25b4_4829
        );
        assert_eq!(
            fnv1a(IQ2XS_GRID.into_iter().flat_map(u64::to_le_bytes)),
            0xc9b1_ee61_e799_09bd
        );
        assert_eq!(
            fnv1a(IQ2S_GRID.into_iter().flat_map(u64::to_le_bytes)),
            0x123e_dd38_a3b6_2b90
        );
        assert_eq!(
            fnv1a(IQ3XXS_GRID.into_iter().flat_map(u32::to_le_bytes)),
            0xdfa5_dc83_d6a1_55d5
        );
        assert_eq!(
            fnv1a(IQ1S_GRID.into_iter().flat_map(u64::to_le_bytes)),
            0x6703_ed86_3501_ae2e
        );
    }

    #[test]
    fn selected_decoders_are_bit_exact_with_scalar_reference() {
        for format in [
            BlockFormat::Mxfp4,
            BlockFormat::Iq4Nl,
            BlockFormat::Iq4Xs,
            BlockFormat::Iq3S,
            BlockFormat::Iq3Xxs,
            BlockFormat::Iq2S,
            BlockFormat::Iq2Xs,
            BlockFormat::Iq2Xxs,
            BlockFormat::Iq1S,
            BlockFormat::Iq1M,
        ] {
            let mut block = vec![0u8; format.block_bytes()];
            for (index, byte) in block.iter_mut().enumerate() {
                *byte = index.wrapping_mul(73).wrapping_add(19) as u8;
            }
            match format {
                BlockFormat::Mxfp4 => block[0] = 128,
                BlockFormat::Iq1M => block[48..56].fill(0),
                _ => block[..2].copy_from_slice(&half::f16::from_f32(0.125).to_le_bytes()),
            }

            let mut scalar = [0.0f32; IQ_SUPER_QK];
            let mut selected = [0.0f32; IQ_SUPER_QK];
            format.scalar_decoder()(&block, &mut scalar[..format.qk()]);
            format.decoder()(&block, &mut selected[..format.qk()]);
            assert_eq!(
                scalar[..format.qk()]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                selected[..format.qk()]
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "{format:?} selected decoder changed f32 bits"
            );
        }
    }

    #[test]
    fn new_iq_formats_register_with_upstream_block_sizes() {
        for (format, block_bytes) in [
            ("iq2_xs", IQ2_XS_BLOCK_BYTES),
            ("iq2_s", IQ2_S_BLOCK_BYTES),
            ("iq3_xxs", IQ3_XXS_BLOCK_BYTES),
            ("iq1_s", IQ1_S_BLOCK_BYTES),
            ("iq1_m", IQ1_M_BLOCK_BYTES),
        ] {
            let (graph, node) = model_node(
                format,
                &[1, IQ_SUPER_QK],
                &[1, 1, block_bytes],
                &[1, 1],
                IQ_SUPER_QK,
                1,
                false,
            );
            let model = Model::new(&graph);
            CpuExecutionProvider::new()
                .get_kernel(model.graph.node(node), &[], 1)
                .expect("implemented IQ format must create a CPU kernel");
        }
    }

    fn packed_smoke_matrix(format: BlockFormat, n: usize) -> (usize, Vec<u8>) {
        let k = format.qk();
        let block_bytes = format.block_bytes();
        let mut packed = vec![0u8; n * block_bytes];
        for output in 0..n {
            let block = &mut packed[output * block_bytes..][..block_bytes];
            match format {
                BlockFormat::Mxfp4 => {
                    block[0] = 127;
                    for (index, byte) in block[1..].iter_mut().enumerate() {
                        *byte = ((index as u8) & 0x0f) | (((15 - index as u8) & 0x0f) << 4);
                    }
                }
                BlockFormat::Iq1M => {}
                _ => block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_le_bytes()),
            }
        }
        (k, packed)
    }

    #[test]
    fn cached_dense_matches_uncached_for_supported_formats_and_activation_dtypes() {
        let n = 2usize;
        for format in [
            BlockFormat::Mxfp4,
            BlockFormat::Iq4Nl,
            BlockFormat::Iq4Xs,
            BlockFormat::Iq3S,
            BlockFormat::Iq3Xxs,
            BlockFormat::Iq2S,
            BlockFormat::Iq2Xs,
            BlockFormat::Iq2Xxs,
            BlockFormat::Iq1S,
            BlockFormat::Iq1M,
        ] {
            let (k, packed) = packed_smoke_matrix(format, n);
            let activations: Vec<f32> = (0..k)
                .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
                .collect();
            let packed_b = Owned::u8(&[n, 1, format.block_bytes()], &packed);
            for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
                let activation = match dtype {
                    DataType::Float32 => Owned::f32(&[1, k], &activations),
                    DataType::Float16 => Owned::f16(&[1, k], &activations),
                    DataType::BFloat16 => Owned::bf16(&[1, k], &activations),
                    _ => unreachable!(),
                };
                let uncached = BlockQuantizedMatMulKernel {
                    k,
                    n,
                    format,
                    packed_b_constant: false,
                    weight_identity: DenseWeightIdentity::default(),
                    weight_cache: DenseWeightCache::new(),
                };
                let mut expected = Owned::zeros(dtype, &[1, n]);
                uncached
                    .execute(
                        &[activation.view(), packed_b.view()],
                        &mut [expected.view_mut()],
                    )
                    .unwrap();

                let mut cached = BlockQuantizedMatMulKernel {
                    k,
                    n,
                    format,
                    packed_b_constant: false,
                    weight_identity: DenseWeightIdentity::default(),
                    weight_cache: DenseWeightCache::new(),
                };
                cached.set_constant_inputs(&[false, true]);
                let hits_before = BLOCK_QUANT_MATMUL_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed);
                let mut actual = Owned::zeros(dtype, &[1, n]);
                cached
                    .execute(
                        &[activation.view(), packed_b.view()],
                        &mut [actual.view_mut()],
                    )
                    .unwrap();
                cached
                    .execute(
                        &[activation.view(), packed_b.view()],
                        &mut [actual.view_mut()],
                    )
                    .unwrap();

                assert_eq!(
                    cached.weight_cache.stats().0,
                    1,
                    "{format:?}/{dtype:?} should reuse one cached dense matrix"
                );
                assert!(
                    BLOCK_QUANT_MATMUL_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed) > hits_before,
                    "{format:?}/{dtype:?} should hit cached-dense on the second call"
                );
                for (index, (actual, expected)) in actual
                    .to_f32()
                    .into_iter()
                    .zip(expected.to_f32())
                    .enumerate()
                {
                    assert!(
                        actual.to_bits() == expected.to_bits() || (actual - expected).abs() <= 1e-5,
                        "{format:?}/{dtype:?} element {index}: cached {actual} != uncached {expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn constant_mxfp4_weight_uses_bounded_cached_dense_path() {
        let (m, k, n) = (2usize, 32usize, 2usize);
        let mut packed = vec![0u8; n * MXFP4_BLOCK_BYTES];
        for output in 0..n {
            let block = &mut packed[output * MXFP4_BLOCK_BYTES..][..MXFP4_BLOCK_BYTES];
            block[0] = 127;
            for byte in &mut block[1..] {
                *byte = if output == 0 { 0x22 } else { 0x33 };
            }
        }
        let activations: Vec<f32> = (0..m * k)
            .map(|index| ((index % 17) as f32 - 8.0) / 8.0)
            .collect();
        let a = Owned::f32(&[m, k], &activations);
        let b = Owned::u8(&[n, 1, MXFP4_BLOCK_BYTES], &packed);
        let mut y = Owned::zeros_f32(&[m, n]);
        let mut kernel = BlockQuantizedMatMulKernel {
            k,
            n,
            format: BlockFormat::Mxfp4,
            packed_b_constant: true,
            weight_identity: DenseWeightIdentity::default(),
            weight_cache: DenseWeightCache::new(),
        };
        kernel.set_constant_inputs(&[false, true]);

        let hits_before = BLOCK_QUANT_MATMUL_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed);
        kernel
            .execute(&[a.view(), b.view()], &mut [y.view_mut()])
            .unwrap();
        let identity_after_first = kernel.weight_identity.stats();
        let activity_after_first = kernel.weight_cache.activity();
        kernel
            .execute(&[a.view(), b.view()], &mut [y.view_mut()])
            .unwrap();

        assert_eq!(
            kernel.weight_cache.stats().0,
            1,
            "constant packed_B should occupy one bounded dense cache entry across repeated calls"
        );
        assert_eq!(
            kernel.weight_identity.stats(),
            identity_after_first,
            "a stable constant cache hit must not copy or hash packed_B again"
        );
        assert_eq!(
            identity_after_first,
            (1, packed.len(), 0),
            "the initial contiguous weight should be hashed once without materialization"
        );
        assert_eq!(
            kernel.weight_cache.activity(),
            (activity_after_first.0 + 1, activity_after_first.1),
            "the repeated call must hit without another dense expansion"
        );
        assert!(
            BLOCK_QUANT_MATMUL_CACHED_DENSE_TEST_HITS.load(Ordering::Relaxed) > hits_before,
            "second execution must prove the cached-dense optimized path"
        );
    }

    #[test]
    fn dense_weight_cache_is_bounded_and_lru_evicts() {
        let cache = DenseWeightCache::with_limit(16);
        let packed1 = Owned::u8(&[4], &[1, 2, 3, 4]);
        let packed2 = Owned::u8(&[4], &[5, 6, 7, 8]);
        let identity1 = DenseWeightIdentity::default();
        let identity2 = DenseWeightIdentity::default();
        let packed1_view = packed1.view();
        let packed2_view = packed2.view();
        let key1 = identity1
            .resolve(&packed1_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&packed1_view)
            })
            .unwrap()
            .key;
        let key2 = identity2
            .resolve(&packed2_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&packed2_view)
            })
            .unwrap()
            .key;
        let _ = cache
            .get_or_insert_with(key1.as_ref(), || Ok(vec![1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        assert_eq!(cache.stats(), (1, 16));
        let (_, status) = cache
            .get_or_insert_with(key1.as_ref(), || unreachable!("cache hit"))
            .unwrap();
        assert_eq!(status, DenseWeightCacheStatus::Hit);
        let _ = cache
            .get_or_insert_with(key2.as_ref(), || Ok(vec![5.0, 6.0, 7.0, 8.0]))
            .unwrap();
        assert_eq!(
            cache.stats(),
            (1, 16),
            "second 16-byte entry must evict the older first entry under a 16-byte bound"
        );
        let (_, status) = cache
            .get_or_insert_with(key1.as_ref(), || Ok(vec![9.0, 10.0, 11.0, 12.0]))
            .unwrap();
        assert_eq!(status, DenseWeightCacheStatus::MissStored);
    }

    #[test]
    fn dense_weight_cache_disabled_and_oversize_entries_are_not_retained() {
        let packed = Owned::u8(&[4], &[1, 2, 3, 4]);
        let identity = DenseWeightIdentity::default();
        let packed_view = packed.view();
        let key = identity
            .resolve(&packed_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&packed_view)
            })
            .unwrap()
            .key;

        for limit in [0, 15] {
            let cache = DenseWeightCache::with_limit(limit);
            for expected_builds in 1..=2 {
                let (_, status) = cache
                    .get_or_insert_with(key.as_ref(), || Ok(vec![1.0, 2.0, 3.0, 4.0]))
                    .unwrap();
                assert_eq!(status, DenseWeightCacheStatus::MissNotStored);
                assert_eq!(cache.stats(), (0, 0));
                assert_eq!(cache.activity(), (0, expected_builds));
            }
        }
    }

    #[test]
    fn dense_weight_identity_rekeys_when_the_constant_source_changes() {
        let first = Owned::u8(&[4], &[1, 2, 3, 4]);
        let second = Owned::u8(&[4], &[4, 3, 2, 1]);
        let identity = DenseWeightIdentity::default();
        let first_view = first.view();
        let first_key = identity
            .resolve(&first_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&first_view)
            })
            .unwrap()
            .key;
        let repeated_key = identity
            .resolve(&first_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                unreachable!("stable identity must not request the payload again")
            })
            .unwrap()
            .key;
        assert!(Arc::ptr_eq(&first_key, &repeated_key));

        let second_view = second.view();
        let second_key = identity
            .resolve(&second_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&second_view)
            })
            .unwrap()
            .key;
        assert_ne!(first_key, second_key);
        assert_eq!(
            identity.stats(),
            (1, 8, 0),
            "each observable source identity must be hashed exactly once"
        );
    }

    #[test]
    fn dense_weight_identity_uses_mmap_owner_metadata_not_just_address() {
        let packed = Owned::u8(&[4], &[1, 2, 3, 4]);
        let identity = DenseWeightIdentity::default();
        let first_view = packed.view().with_backing(TensorBacking::ExternalMmap(
            onnx_runtime_ep_api::ExternalMmapRegion {
                mapping_id: 41,
                offset: 128,
                len: 4,
            },
        ));
        let first_key = identity
            .resolve(&first_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&first_view)
            })
            .unwrap()
            .key;
        let second_view = packed.view().with_backing(TensorBacking::ExternalMmap(
            onnx_runtime_ep_api::ExternalMmapRegion {
                mapping_id: 42,
                offset: 128,
                len: 4,
            },
        ));
        let second_key = identity
            .resolve(&second_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&second_view)
            })
            .unwrap()
            .key;
        assert_ne!(first_key, second_key);
        assert_eq!(identity.stats(), (1, 8, 0));
    }

    #[test]
    fn dense_weight_cache_env_parser_handles_disable_whitespace_and_overflow() {
        assert_eq!(parse_dense_weight_cache_bytes(Some("0")), 0);
        assert_eq!(parse_dense_weight_cache_bytes(Some(" 4096 ")), 4096);
        assert_eq!(
            parse_dense_weight_cache_bytes(Some("184467440737095516160")),
            DEFAULT_DENSE_WEIGHT_CACHE_BYTES
        );
        assert_eq!(
            parse_dense_weight_cache_bytes(Some("not-a-number")),
            DEFAULT_DENSE_WEIGHT_CACHE_BYTES
        );
        assert_eq!(
            parse_dense_weight_cache_bytes(None),
            DEFAULT_DENSE_WEIGHT_CACHE_BYTES
        );
    }

    #[test]
    fn dense_weight_cache_concurrent_miss_builds_once() {
        use std::sync::Barrier;

        let cache = std::sync::Arc::new(DenseWeightCache::with_limit(1024));
        let packed = Owned::u8(&[4], &[9, 8, 7, 6]);
        let identity = DenseWeightIdentity::default();
        let packed_view = packed.view();
        let key = identity
            .resolve(&packed_view, BlockFormat::Mxfp4, 4, 1, 0, None, || {
                packed_tensor_bytes(&packed_view)
            })
            .unwrap()
            .key;
        let builds = std::sync::Arc::new(AtomicUsize::new(0));
        let barrier = std::sync::Arc::new(Barrier::new(4));
        std::thread::scope(|scope| {
            for _ in 0..4 {
                let cache = std::sync::Arc::clone(&cache);
                let builds = std::sync::Arc::clone(&builds);
                let barrier = std::sync::Arc::clone(&barrier);
                let key = Arc::clone(&key);
                scope.spawn(move || {
                    barrier.wait();
                    let _ = cache
                        .get_or_insert_with(key.as_ref(), || {
                            builds.fetch_add(1, Ordering::Relaxed);
                            Ok(vec![1.0, 2.0, 3.0, 4.0])
                        })
                        .unwrap();
                });
            }
        });
        assert_eq!(
            builds.load(Ordering::Relaxed),
            1,
            "concurrent cache miss for the same immutable weight must build once"
        );
    }

    #[test]
    #[ignore = "representative CPU throughput benchmark; run with --release -- --ignored"]
    fn benchmark_prefill_4096x4096_m64() {
        use std::hint::black_box;
        use std::time::Instant;

        const M: usize = 64;
        const K: usize = 4096;
        const N: usize = 4096;

        for format in [BlockFormat::Mxfp4, BlockFormat::Iq4Nl] {
            let block_bytes = format.block_bytes();
            let blocks = K.div_ceil(format.qk());
            let mut packed = vec![0u8; N * blocks * block_bytes];
            for (index, block) in packed.chunks_exact_mut(block_bytes).enumerate() {
                match format {
                    BlockFormat::Mxfp4 => block[0] = 128,
                    BlockFormat::Iq4Nl => {
                        block[..2].copy_from_slice(&half::f16::from_f32(0.01).to_le_bytes());
                    }
                    _ => unreachable!(),
                }
                for (offset, byte) in block[block_bytes - format.qk() / 2..]
                    .iter_mut()
                    .enumerate()
                {
                    *byte = index.wrapping_mul(17).wrapping_add(offset * 29) as u8;
                }
            }
            let packed = Owned::u8(&[N, blocks, block_bytes], &packed);
            let kernel = BlockQuantizedMatMulKernel {
                k: K,
                n: N,
                format,
                packed_b_constant: false,
                weight_identity: DenseWeightIdentity::default(),
                weight_cache: DenseWeightCache::new(),
            };

            let decode_start = Instant::now();
            let weight = black_box(kernel.dequantize_weight_kn(&packed.view()).unwrap());
            let decode = decode_start.elapsed();
            let activations: Vec<f32> = (0..M * K)
                .map(|index| (index % 31) as f32 * (1.0 / 31.0))
                .collect();
            let mut result = vec![0.0f32; M * N];
            let gemm_start = Instant::now();
            gemm(
                black_box(&activations),
                black_box(&weight),
                black_box(&mut result),
                M,
                K,
                N,
            )
            .unwrap();
            let gemm_time = gemm_start.elapsed();
            let gflops = 2.0 * M as f64 * K as f64 * N as f64 / gemm_time.as_secs_f64() / 1.0e9;
            eprintln!(
                "{format:?}: decode={:.1} ms prefill={:.1} ms ({gflops:.1} GFLOP/s)",
                decode.as_secs_f64() * 1.0e3,
                gemm_time.as_secs_f64() * 1.0e3,
            );
            black_box(result);
        }
    }
}
