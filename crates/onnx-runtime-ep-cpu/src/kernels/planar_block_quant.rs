//! Planar block-scaled quantized weight formats for DeepSeek-V4 (Mobius #602 B2).
//!
//! Two native checkpoint layouts that the interleaved single-tensor
//! [`BlockFormat`](super::block_quantized_matmul::BlockFormat) family cannot
//! represent, because each stores its block scales in a **separate** tensor (an
//! auxiliary scale bank) rather than inline with the packed weight bytes:
//!
//! 1. **block-FP8** ([`PlanarBlockFormat::BlockFp8`]) — an `F8_E4M3` weight of
//!    *logical* shape `[out, in]` (E4M3 is one byte per element, never sub-byte
//!    packed) paired with an `F8_E8M0` (UE8M0, exponent-only, power-of-two) 2D
//!    block scale of shape `[ceil(out / bs0), ceil(in / bs1)]`. DeepSeek-V4's
//!    attention projections and shared experts use `bs0 = bs1 = 128`.
//!
//! 2. **planar-FP4** ([`PlanarBlockFormat::Fp4Planar`]) — an `I8` weight that
//!    stores **two `E2M1` nibbles per byte** (packed shape `[out, in / 2]`,
//!    logical `[out, in]`) paired with an `F8_E8M0` 1D micro-scale of shape
//!    `[out, in / 32]` (one UE8M0 exponent per output row per 32 logical input
//!    elements). Numerically this is **MXFP4** (E2M1 + block-32 + E8M0), but the
//!    byte layout is *planar* — the nibbles and the scales live in two distinct
//!    tensors, unlike the interleaved llama.cpp `block_mxfp4` (`QK=32`, 17
//!    bytes/block = 1 E8M0 byte + 16 nibble bytes) that
//!    [`BlockFormat::Mxfp4`](super::block_quantized_matmul::BlockFormat::Mxfp4)
//!    decodes. This module deliberately does **not** transcode planar into
//!    interleaved; it decodes the planar bytes directly.
//!
//! The value math (`E4M3`, `E2M1`, `UE8M0`) reuses the vetted primitives in
//! [`super::block_dequant`]. This module adds the planar *layout* contract: a
//! property-typed [`PlanarLayout`] descriptor (logical vs packed shape, scale
//! grid vs block geometry, byte counts) that fails closed on any inconsistency,
//! a CPU dequantization oracle that materialises the dense `[K, N]` (`in`-major)
//! weight, a straight matmul oracle over it, and a byte-exact routed-expert bank
//! ([`PlanarExpertBank`]). All arithmetic is overflow-checked and every decode
//! is fail-closed on reserved codes — never a silent dequantize-to-float copy,
//! never a dense-expert fallback.

use onnx_runtime_ep_api::{EpError, Result};

use super::block_dequant::{decode_e2m1, decode_e4m3fn, decode_e8m0_scale};

/// Logical input elements per UE8M0 micro-scale in a planar-FP4 tensor. MXFP4
/// pins this to 32 (NVFP4 would instead use 16 + an E4M3 block scale + an FP32
/// global scale, which this format is explicitly not).
pub const FP4_MICROSCALE_BLOCK: usize = 32;

/// Two E2M1 nibbles are packed into every `I8` byte of a planar-FP4 weight.
pub const FP4_PACK_FACTOR: usize = 2;

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("planar block quant: {}", message.into()))
}

/// Stable content identity returned by fail-closed planar bank admission.
///
/// The identity is process-independent and covers the format, logical/block
/// geometry, packed bytes, and scale bytes. CUDA keeps this result with its
/// immutable weight-bank admission so graph replay never rescans host data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlanarBankIdentity(u64);

impl PlanarBankIdentity {
    pub fn get(self) -> u64 {
        self.0
    }
}

const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn hash_bytes(mut hash: u64, bytes: &[u8]) -> u64 {
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

fn hash_u64(hash: u64, value: u64) -> u64 {
    hash_bytes(hash, &value.to_le_bytes())
}

fn bank_identity(
    layout: &PlanarLayout,
    num_experts: usize,
    packed: &[u8],
    scale: &[u8],
) -> PlanarBankIdentity {
    let mut hash = FNV_OFFSET_BASIS;
    hash = hash_u64(
        hash,
        match layout.format {
            PlanarBlockFormat::BlockFp8 => 0,
            PlanarBlockFormat::Fp4Planar => 1,
        },
    );
    for value in [
        layout.out_features,
        layout.in_features,
        layout.block_out,
        layout.block_in,
        num_experts,
    ] {
        hash = hash_u64(hash, value as u64);
    }
    hash = hash_bytes(hash, packed);
    hash = hash_bytes(hash, scale);
    PlanarBankIdentity(hash)
}

// ---------------------------------------------------------------------------
// Format
// ---------------------------------------------------------------------------

/// A planar block-scaled weight format: a packed weight tensor plus a separate
/// UE8M0 scale bank. Distinct from the interleaved single-tensor
/// [`BlockFormat`](super::block_quantized_matmul::BlockFormat) family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlanarBlockFormat {
    /// `F8_E4M3` weight + 2D `F8_E8M0` block scale.
    BlockFp8,
    /// `I8`-packed `E2M1` nibbles + 1D `F8_E8M0` block-32 micro-scale.
    Fp4Planar,
}

impl PlanarBlockFormat {
    /// Parse the runtime format string. These names are the runtime capability
    /// strings the Mobius #602 / Deckard #593 emitters target; they are
    /// deliberately distinct from the interleaved `mxfp4` name.
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "block_fp8" => Ok(Self::BlockFp8),
            "fp4_planar" => Ok(Self::Fp4Planar),
            _ => Err(error(format!(
                "unsupported planar format '{value}'; supported planar formats are block_fp8 and fp4_planar"
            ))),
        }
    }

    /// The stable runtime capability / format string for this format.
    pub fn capability_str(self) -> &'static str {
        match self {
            Self::BlockFp8 => "block_fp8",
            Self::Fp4Planar => "fp4_planar",
        }
    }

    /// Name of the packed-weight element dtype, as it appears in a safetensors
    /// header.
    pub fn weight_dtype_name(self) -> &'static str {
        match self {
            Self::BlockFp8 => "F8_E4M3",
            Self::Fp4Planar => "I8",
        }
    }

    /// Name of the scale-bank element dtype (both formats use UE8M0).
    pub fn scale_dtype_name(self) -> &'static str {
        "F8_E8M0"
    }

    /// Number of logical weight elements stored per packed byte along the input
    /// dimension: 1 for block-FP8 (E4M3 is one byte each), 2 for planar-FP4
    /// (two E2M1 nibbles per byte).
    pub fn pack_factor(self) -> usize {
        match self {
            Self::BlockFp8 => 1,
            Self::Fp4Planar => FP4_PACK_FACTOR,
        }
    }
}

// ---------------------------------------------------------------------------
// Layout / bank descriptor (property-typed contract)
// ---------------------------------------------------------------------------

/// The property-typed layout contract for one planar projection: logical
/// `[out, in]` geometry, block dimensions, and every derived packed/scale
/// shape + byte count. Constructed via [`PlanarLayout::new`], which fails closed
/// on any geometric inconsistency, so a constructed value is the single source
/// of offset truth shared by the validator, the CPU oracle, and (once it can
/// prove parity) the CUDA claim gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanarLayout {
    format: PlanarBlockFormat,
    out_features: usize,
    in_features: usize,
    /// Output rows covered by one scale (1 for planar-FP4).
    block_out: usize,
    /// Logical input elements covered by one scale (32 for planar-FP4).
    block_in: usize,
}

impl PlanarLayout {
    /// Build and validate a planar layout for a logical `[out, in]` projection.
    ///
    /// * `block_fp8` requires a positive 2D `(block_out, block_in)` (DeepSeek-V4
    ///   uses `(128, 128)`).
    /// * `fp4_planar` pins the block to `(1, 32)`; `in` must be even (two
    ///   nibbles per byte) and divisible by 32 (one micro-scale per 32 inputs).
    pub fn new(
        format: PlanarBlockFormat,
        out_features: usize,
        in_features: usize,
        block_out: usize,
        block_in: usize,
    ) -> Result<Self> {
        if out_features == 0 || in_features == 0 {
            return Err(error(format!(
                "logical shape [{out_features}, {in_features}] must be non-empty"
            )));
        }
        match format {
            PlanarBlockFormat::BlockFp8 => {
                if block_out == 0 || block_in == 0 {
                    return Err(error(format!(
                        "block_fp8 needs a positive 2D block, got [{block_out}, {block_in}]"
                    )));
                }
            }
            PlanarBlockFormat::Fp4Planar => {
                if block_out != 1 || block_in != FP4_MICROSCALE_BLOCK {
                    return Err(error(format!(
                        "fp4_planar block must be [1, {FP4_MICROSCALE_BLOCK}], got [{block_out}, {block_in}]"
                    )));
                }
                if !in_features.is_multiple_of(FP4_PACK_FACTOR) {
                    return Err(error(format!(
                        "fp4_planar logical in={in_features} must be even (two E2M1 nibbles per byte)"
                    )));
                }
                if !in_features.is_multiple_of(FP4_MICROSCALE_BLOCK) {
                    return Err(error(format!(
                        "fp4_planar logical in={in_features} must be divisible by micro-scale block {FP4_MICROSCALE_BLOCK}"
                    )));
                }
            }
        }
        Ok(Self {
            format,
            out_features,
            in_features,
            block_out,
            block_in,
        })
    }

    pub fn format(&self) -> PlanarBlockFormat {
        self.format
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn block_shape(&self) -> (usize, usize) {
        (self.block_out, self.block_in)
    }

    /// Packed-weight shape `[out, in / pack_factor]` (planar-FP4 packs two
    /// nibbles per byte; block-FP8 is 1:1 so packed == logical).
    pub fn packed_shape(&self) -> [usize; 2] {
        [
            self.out_features,
            self.in_features / self.format.pack_factor(),
        ]
    }

    /// Number of scale rows: `ceil(out / block_out)`.
    pub fn scale_rows(&self) -> usize {
        self.out_features.div_ceil(self.block_out)
    }

    /// Number of scale columns: `ceil(in / block_in)`.
    pub fn scale_cols(&self) -> usize {
        self.in_features.div_ceil(self.block_in)
    }

    /// UE8M0 scale-bank shape `[scale_rows, scale_cols]`.
    pub fn scale_shape(&self) -> [usize; 2] {
        [self.scale_rows(), self.scale_cols()]
    }

    /// Total packed-weight byte count (one byte per packed element for both
    /// formats). Overflow-checked.
    pub fn packed_bytes(&self) -> Result<usize> {
        let [rows, cols] = self.packed_shape();
        rows.checked_mul(cols)
            .ok_or_else(|| error("packed byte count overflow"))
    }

    /// Total scale-bank byte count (one UE8M0 byte per scale). Overflow-checked.
    pub fn scale_bytes(&self) -> Result<usize> {
        self.scale_rows()
            .checked_mul(self.scale_cols())
            .ok_or_else(|| error("scale byte count overflow"))
    }

    /// Number of dequantized `[K, N]` weight elements (`in * out`).
    /// Overflow-checked.
    pub fn dense_elements(&self) -> Result<usize> {
        self.in_features
            .checked_mul(self.out_features)
            .ok_or_else(|| error("dense weight element count overflow"))
    }

    /// Fail-closed validation of a concrete `(packed, scale)` tensor pair
    /// against this layout: dtype names, both shapes, and exact byte counts.
    /// Any mismatch is a typed rejection — never a dequantize-to-float fallback.
    pub fn validate_tensors(
        &self,
        packed_dtype: &str,
        packed_shape: &[usize],
        packed_len: usize,
        scale_dtype: &str,
        scale_shape: &[usize],
        scale_len: usize,
    ) -> Result<()> {
        if packed_dtype != self.format.weight_dtype_name() {
            return Err(error(format!(
                "{} weight must be {}, got {packed_dtype}",
                self.format.capability_str(),
                self.format.weight_dtype_name()
            )));
        }
        if scale_dtype != self.format.scale_dtype_name() {
            return Err(error(format!(
                "{} scale must be {} (UE8M0), got {scale_dtype}",
                self.format.capability_str(),
                self.format.scale_dtype_name()
            )));
        }
        if packed_shape != self.packed_shape() {
            return Err(error(format!(
                "{} packed shape {:?} != expected {:?} (logical [{}, {}], pack factor {})",
                self.format.capability_str(),
                packed_shape,
                self.packed_shape(),
                self.out_features,
                self.in_features,
                self.format.pack_factor()
            )));
        }
        if scale_shape != self.scale_shape() {
            return Err(error(format!(
                "{} scale grid {:?} != expected {:?} (ceil(out/{}), ceil(in/{}))",
                self.format.capability_str(),
                scale_shape,
                self.scale_shape(),
                self.block_out,
                self.block_in
            )));
        }
        let expected_packed = self.packed_bytes()?;
        if packed_len != expected_packed {
            return Err(error(format!(
                "{} packed weight must be {expected_packed} bytes, got {packed_len}",
                self.format.capability_str()
            )));
        }
        let expected_scale = self.scale_bytes()?;
        if scale_len != expected_scale {
            return Err(error(format!(
                "{} scale bank must be {expected_scale} bytes, got {scale_len}",
                self.format.capability_str()
            )));
        }
        Ok(())
    }

    /// The UE8M0 scale byte index for logical element `(out_row, in_col)`.
    #[inline]
    fn scale_index(&self, out_row: usize, in_col: usize) -> usize {
        (out_row / self.block_out) * self.scale_cols() + (in_col / self.block_in)
    }
}

// ---------------------------------------------------------------------------
// CPU dequantization oracle: dense [K = in, N = out] weight
// ---------------------------------------------------------------------------

/// Decode one logical weight element `(out_row, in_col)` from the planar bytes.
/// Fail-closed on reserved E8M0 / E4M3 codes and on any non-finite product.
#[inline]
fn decode_element(
    layout: &PlanarLayout,
    packed: &[u8],
    scale: &[u8],
    out_row: usize,
    in_col: usize,
) -> Result<f32> {
    let scale_exponent = scale[layout.scale_index(out_row, in_col)];
    let scale_value = decode_e8m0_scale(scale_exponent);
    if !scale_value.is_finite() {
        return Err(error(format!(
            "{} uses reserved E8M0 scale exponent 0x{scale_exponent:02x}",
            layout.format.capability_str()
        )));
    }
    let code_value = match layout.format {
        PlanarBlockFormat::BlockFp8 => {
            let code = packed[out_row * layout.in_features + in_col];
            let value = decode_e4m3fn(code);
            if !value.is_finite() {
                return Err(error(format!(
                    "block_fp8 weight contains reserved E4M3 NaN code 0x{code:02x}"
                )));
            }
            value
        }
        PlanarBlockFormat::Fp4Planar => {
            let byte = packed[out_row * (layout.in_features / FP4_PACK_FACTOR) + in_col / 2];
            let nibble = if in_col.is_multiple_of(2) {
                byte & 0x0f
            } else {
                byte >> 4
            };
            decode_e2m1(nibble)
        }
    };
    let result = code_value * scale_value;
    if !result.is_finite() {
        return Err(error(format!(
            "{} value {code_value} overflows with block scale {scale_value}",
            layout.format.capability_str()
        )));
    }
    Ok(result)
}

/// Validate every encoded value in one immutable planar weight/scale pair.
///
/// This is the CPU authority for CUDA admission. Exact rejected encodings are:
/// E4M3FN `0x7f` and `0xff`, UE8M0 `0xff`, and any otherwise-finite
/// code×scale pair whose decoded `f32` product overflows. Every E2M1 nibble
/// `0x0..=0xf` is valid; only its product with the selected UE8M0 scale can be
/// rejected. The returned identity caches that successful result for later
/// launches without another scan.
pub fn validate_planar_values(
    layout: &PlanarLayout,
    packed: &[u8],
    scale: &[u8],
) -> Result<PlanarBankIdentity> {
    validate_planar_elements(layout, packed, scale)?;
    Ok(bank_identity(layout, 1, packed, scale))
}

fn validate_planar_elements(layout: &PlanarLayout, packed: &[u8], scale: &[u8]) -> Result<()> {
    require_planar_value_lengths(layout, packed, scale)?;
    for out_row in 0..layout.out_features {
        for in_col in 0..layout.in_features {
            decode_element(layout, packed, scale, out_row, in_col)?;
        }
    }
    Ok(())
}

fn require_planar_value_lengths(layout: &PlanarLayout, packed: &[u8], scale: &[u8]) -> Result<()> {
    if packed.len() != layout.packed_bytes()? {
        return Err(error(format!(
            "{} packed weight must be {} bytes, got {}",
            layout.format.capability_str(),
            layout.packed_bytes()?,
            packed.len()
        )));
    }
    if scale.len() != layout.scale_bytes()? {
        return Err(error(format!(
            "{} scale bank must be {} bytes, got {}",
            layout.format.capability_str(),
            layout.scale_bytes()?,
            scale.len()
        )));
    }
    Ok(())
}

/// Validate an expert-major immutable bank using the same element contract as
/// [`validate_planar_values`].
pub fn validate_planar_expert_bank_values(
    layout: &PlanarLayout,
    num_experts: usize,
    packed_bank: &[u8],
    scale_bank: &[u8],
) -> Result<PlanarBankIdentity> {
    if num_experts == 0 {
        return Err(error("cannot validate an empty routed-expert bank"));
    }
    let per_packed = layout.packed_bytes()?;
    let per_scale = layout.scale_bytes()?;
    let expected_packed = num_experts
        .checked_mul(per_packed)
        .ok_or_else(|| error("packed bank size overflow"))?;
    let expected_scale = num_experts
        .checked_mul(per_scale)
        .ok_or_else(|| error("scale bank size overflow"))?;
    if packed_bank.len() != expected_packed {
        return Err(error(format!(
            "packed expert bank must be experts*{per_packed} = {expected_packed} bytes, got {}",
            packed_bank.len()
        )));
    }
    if scale_bank.len() != expected_scale {
        return Err(error(format!(
            "scale expert bank must be experts*{per_scale} = {expected_scale} bytes, got {}",
            scale_bank.len()
        )));
    }
    for expert in 0..num_experts {
        let packed = &packed_bank[expert * per_packed..][..per_packed];
        let scale = &scale_bank[expert * per_scale..][..per_scale];
        validate_planar_elements(layout, packed, scale)
            .map_err(|err| error(format!("expert {expert} failed value validation: {err}")))?;
    }
    Ok(bank_identity(layout, num_experts, packed_bank, scale_bank))
}

/// Materialise the dense weight in `[K = in, N = out]` (`in`-major) layout, the
/// orientation the runtime matmul consumes (`C[M, N] = A[M, K] * W[K, N]`).
/// This is the CPU oracle. `packed` and `scale` must already have passed
/// [`PlanarLayout::validate_tensors`]; their lengths are re-checked here so the
/// function is memory-safe on any input.
pub fn dequantize_planar_kn(
    layout: &PlanarLayout,
    packed: &[u8],
    scale: &[u8],
) -> Result<Vec<f32>> {
    require_planar_value_lengths(layout, packed, scale)?;
    let n = layout.out_features;
    let k = layout.in_features;
    let mut weight_kn = vec![0.0f32; layout.dense_elements()?];
    for out_row in 0..n {
        for in_col in 0..k {
            weight_kn[in_col * n + out_row] =
                decode_element(layout, packed, scale, out_row, in_col)?;
        }
    }
    Ok(weight_kn)
}

/// Matmul oracle: `C[M, N] = A[M, K] * W[K, N]`, where `W` is decoded from the
/// planar `(packed, scale)` bytes. A straight triple loop over the dense
/// weight — the correctness reference the CUDA kernel must match before it may
/// claim a planar node. `a` is row-major `[m_rows, in]`.
pub fn planar_block_matmul(
    a: &[f32],
    m_rows: usize,
    layout: &PlanarLayout,
    packed: &[u8],
    scale: &[u8],
) -> Result<Vec<f32>> {
    let k = layout.in_features;
    let n = layout.out_features;
    if a.len()
        != m_rows
            .checked_mul(k)
            .ok_or_else(|| error("A element count overflow"))?
    {
        return Err(error(format!(
            "A must be [{m_rows}, {k}] = {} elements, got {}",
            m_rows * k,
            a.len()
        )));
    }
    let weight_kn = dequantize_planar_kn(layout, packed, scale)?;
    let mut out = vec![
        0.0f32;
        m_rows
            .checked_mul(n)
            .ok_or_else(|| error("C element count overflow"))?
    ];
    for row in 0..m_rows {
        let a_row = &a[row * k..][..k];
        let c_row = &mut out[row * n..][..n];
        for (col, weight_col) in weight_kn.chunks_exact(n).enumerate() {
            let a_val = a_row[col];
            if a_val == 0.0 {
                continue;
            }
            for (acc, &w) in c_row.iter_mut().zip(weight_col) {
                *acc += a_val * w;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Routed-expert bank descriptor (byte-exact, expert-major)
// ---------------------------------------------------------------------------

/// A byte-exact expert-major bank of planar routed-expert weights: every expert
/// shares one [`PlanarLayout`], and its packed bytes + scale bytes are stored
/// contiguously (expert 0's payload, then expert 1's, ...). The per-expert
/// bytes are recoverable by slicing — no re-quantization, no re-ordering. This
/// is the reusable primitive the `BlockQuantizedMoE` routed-expert path (and a
/// future native bank emitter) consumes. A ragged bank is a hard error.
#[derive(Clone, Debug)]
pub struct PlanarExpertBank {
    layout: PlanarLayout,
    num_experts: usize,
    packed_bank: Vec<u8>,
    scale_bank: Vec<u8>,
}

impl PlanarExpertBank {
    /// Assemble a bank from `num_experts` equal-length `(packed, scale)`
    /// payloads that all share `layout`. Each expert is validated against the
    /// layout; a wrong-length payload is a typed rejection.
    pub fn stack(layout: PlanarLayout, per_expert: &[(&[u8], &[u8])]) -> Result<Self> {
        if per_expert.is_empty() {
            return Err(error("cannot stack an empty routed-expert bank"));
        }
        let packed_len = layout.packed_bytes()?;
        let scale_len = layout.scale_bytes()?;
        let mut packed_bank = Vec::with_capacity(
            per_expert
                .len()
                .checked_mul(packed_len)
                .ok_or_else(|| error("packed bank size overflow"))?,
        );
        let mut scale_bank = Vec::with_capacity(
            per_expert
                .len()
                .checked_mul(scale_len)
                .ok_or_else(|| error("scale bank size overflow"))?,
        );
        for (expert, (packed, scale)) in per_expert.iter().enumerate() {
            if packed.len() != packed_len {
                return Err(error(format!(
                    "ragged expert bank: expert {expert} packed weight has {} bytes, expected {packed_len}",
                    packed.len()
                )));
            }
            if scale.len() != scale_len {
                return Err(error(format!(
                    "ragged expert bank: expert {expert} scale has {} bytes, expected {scale_len}",
                    scale.len()
                )));
            }
            packed_bank.extend_from_slice(packed);
            scale_bank.extend_from_slice(scale);
        }
        Ok(Self {
            layout,
            num_experts: per_expert.len(),
            packed_bank,
            scale_bank,
        })
    }

    pub fn layout(&self) -> &PlanarLayout {
        &self.layout
    }

    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Byte-exact packed weight slice for expert `expert`.
    pub fn expert_packed(&self, expert: usize) -> Result<&[u8]> {
        if expert >= self.num_experts {
            return Err(error(format!(
                "expert {expert} out of range [0, {})",
                self.num_experts
            )));
        }
        let len = self.layout.packed_bytes()?;
        Ok(&self.packed_bank[expert * len..][..len])
    }

    /// Byte-exact scale slice for expert `expert`.
    pub fn expert_scale(&self, expert: usize) -> Result<&[u8]> {
        if expert >= self.num_experts {
            return Err(error(format!(
                "expert {expert} out of range [0, {})",
                self.num_experts
            )));
        }
        let len = self.layout.scale_bytes()?;
        Ok(&self.scale_bank[expert * len..][..len])
    }

    /// Dequantize expert `expert` to its dense `[K, N]` weight via the oracle.
    pub fn dequantize_expert_kn(&self, expert: usize) -> Result<Vec<f32>> {
        dequantize_planar_kn(
            &self.layout,
            self.expert_packed(expert)?,
            self.expert_scale(expert)?,
        )
    }
}

#[cfg(test)]
mod tests;
