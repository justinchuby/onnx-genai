//! Shared explicit packed-projection arithmetic and caller-lane demand.
use super::block_quantized_matmul::{BlockFormat, linear_packed_prepared, projection_plan};
use super::decoded_projection::{DecodedProjectionPlan, ProjectionScratch, rayon_decoded_columns};
use onnx_runtime_ep_api::{EpError, Result, WorkspaceLayout};
use onnx_runtime_ir::packed_arithmetic::ActivationArithmetic;

const QK: usize = 32;
const ACTIVATION_BYTES: usize = 34;
fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(message.into())
}

#[derive(Clone, Copy, Debug)]
pub(super) enum PackedProjectionPlan {
    Float32(DecodedProjectionPlan),
    Q8_0,
}
impl PackedProjectionPlan {
    pub(super) fn for_current_pool(arithmetic: ActivationArithmetic) -> Self {
        match arithmetic {
            ActivationArithmetic::Float32 => Self::Float32(projection_plan()),
            ActivationArithmetic::Q8_0 => Self::Q8_0,
        }
    }
    pub(super) fn layout(self, rows: usize, k: usize, n: usize) -> Result<WorkspaceLayout<2>> {
        match self {
            Self::Float32(plan) => plan.layout(rows, k, n),
            Self::Q8_0 => {
                let (columns, bytes) = q8_geometry(rows, k, n)?;
                let invalid = || {
                    error(format!(
                        "q8_0 projection workspace exceeds addressable capacity for rows={rows} K={k} N={n}; reduce projection dimensions"
                    ))
                };
                WorkspaceLayout::new([
                    std::alloc::Layout::array::<f32>(columns).map_err(|_| invalid())?,
                    std::alloc::Layout::array::<f32>(bytes.div_ceil(4)).map_err(|_| invalid())?,
                ])
            }
        }
    }
    pub(super) fn maximum_layout(
        self,
        rows: usize,
        projections: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<std::alloc::Layout> {
        match self {
            Self::Float32(plan) => plan.maximum_layout(rows, projections),
            Self::Q8_0 => {
                let mut bytes = 0;
                let mut alignment = 1;
                for (k, n) in projections {
                    let layout = self.layout(rows, k, n)?;
                    bytes = bytes.max(layout.bytes());
                    alignment = alignment.max(layout.alignment());
                }
                std::alloc::Layout::from_size_align(bytes,alignment).map_err(|_|error("q8_0 serial projection workspace exceeds addressable capacity; reduce batch dimensions"))
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn project(
        self,
        format: BlockFormat,
        packed: &[u8],
        input: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
        scratch: &mut ProjectionScratch<'_>,
        output: &mut [f32],
    ) -> Result<()> {
        match self {
            Self::Float32(plan) => linear_packed_prepared(
                format, packed, input, rows, k, n, bias, plan, scratch, output,
            ),
            Self::Q8_0 => project_q8(format, packed, input, rows, k, n, bias, scratch, output),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum DotFormat {
    Q8_0,
    Q4_0,
}
impl DotFormat {
    fn from_format(format: BlockFormat) -> Result<Self> {
        use onnx_runtime_ir::dequantize_packed::Format;
        match format {
            BlockFormat::Q8_0 | BlockFormat::Onnx(Format::Q8_0) => Ok(Self::Q8_0),
            BlockFormat::Onnx(Format::Q4_0) => Ok(Self::Q4_0),
            _ => Err(error(format!(
                "activation_arithmetic='q8_0' does not implement weight format {format:?}; use Q8_0 or gguf:q4_0 weights, or select activation_arithmetic='float32' with block_layout_version=1"
            ))),
        }
    }
    fn bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
        }
    }
    fn dot(self, w: &[u8], a: &[u8]) -> i32 {
        match self {
            Self::Q8_0 => w[2..]
                .iter()
                .zip(&a[2..])
                .map(|(&w, &a)| i32::from(w as i8) * i32::from(a as i8))
                .sum(),
            Self::Q4_0 => w[2..]
                .iter()
                .enumerate()
                .map(|(i, &w)| {
                    (i32::from(w & 15) - 8) * i32::from(a[2 + i] as i8)
                        + (i32::from(w >> 4) - 8) * i32::from(a[18 + i] as i8)
                })
                .sum(),
        }
    }
}
pub(super) fn validate_format(arithmetic: ActivationArithmetic, format: BlockFormat) -> Result<()> {
    if arithmetic == ActivationArithmetic::Q8_0 {
        DotFormat::from_format(format)?;
    }
    Ok(())
}
fn q8_geometry(rows: usize, k: usize, n: usize) -> Result<(usize, usize)> {
    if rows == 0 || n == 0 {
        return Ok((0, 0));
    }
    let invalid = || {
        error(format!(
            "q8_0 projection dimensions rows={rows}, K={k}, N={n} overflow workspace sizing; reduce batch or projection dimensions"
        ))
    };
    let columns = if rows > 1 && n > 1 {
        rows.checked_mul(n).ok_or_else(invalid)?
    } else {
        0
    };
    let bytes = rows
        .checked_mul(k.div_ceil(QK))
        .and_then(|n| n.checked_mul(ACTIVATION_BYTES))
        .ok_or_else(invalid)?;
    Ok((columns, bytes))
}

/// Publishes a packed view only after every activation block was overwritten,
/// including partial-K padding. Failures leave scratch private to this call.
fn prepare_activations<'a>(
    input: &[f32],
    rows: usize,
    k: usize,
    bytes: &'a mut [u8],
) -> Result<&'a [u8]> {
    let blocks = k.div_ceil(QK);
    for row in 0..rows {
        for block in 0..blocks {
            let start = row * k + block * QK;
            let values = &input[start..(start + QK).min((row + 1) * k)];
            let mut maximum = 0f32;
            for &value in values {
                if !value.is_finite() {
                    return Err(error(format!(
                        "q8_0 projection activation grouped row {row} block {block} contains non-finite values; select activation_arithmetic='float32' with block_layout_version=1"
                    )));
                }
                maximum = maximum.max(value.abs());
            }
            let scale = maximum / 127.;
            let stored = half::f16::from_f32(scale);
            if !stored.is_finite() || (maximum != 0. && stored == half::f16::ZERO) {
                return Err(error(format!(
                    "q8_0 projection activation grouped row {row} block {block} scale {scale} cannot be stored as finite nonzero f16; select activation_arithmetic='float32' with block_layout_version=1"
                )));
            }
            let target = &mut bytes[(row * blocks + block) * ACTIVATION_BYTES
                ..(row * blocks + block + 1) * ACTIVATION_BYTES];
            target[..2].copy_from_slice(&stored.to_le_bytes());
            target[2..].fill(0);
            if maximum != 0. {
                for (target, &value) in target[2..].iter_mut().zip(values) {
                    *target = ((value / scale).round().clamp(-127., 127.) as i8) as u8;
                }
            }
            #[cfg(test)]
            ACTIVATION_CONVERSIONS.with(|v| v.set(v.get() + 1));
        }
    }
    Ok(bytes)
}

#[allow(clippy::too_many_arguments)]
fn project_q8(
    format: BlockFormat,
    packed: &[u8],
    input: &[f32],
    rows: usize,
    k: usize,
    n: usize,
    bias: Option<&[f32]>,
    scratch: &mut ProjectionScratch<'_>,
    output: &mut [f32],
) -> Result<()> {
    let format = DotFormat::from_format(format)?;
    let blocks = k.div_ceil(QK);
    let stride = blocks
        .checked_mul(format.bytes())
        .ok_or_else(|| error("q8_0 packed weight row byte count overflows; reduce K"))?;
    if rows.checked_mul(k) != Some(input.len())
        || n.checked_mul(stride) != Some(packed.len())
        || rows.checked_mul(n) != Some(output.len())
        || bias.is_some_and(|b| b.len() != n)
    {
        return Err(error(format!(
            "q8_0 projection storage does not match rows={rows}, K={k}, N={n}; supply exact activation, packed weight, output and bias lengths"
        )));
    }
    let (column_count, activation_bytes) = q8_geometry(rows, k, n)?;
    let (column_major, activation_storage) =
        scratch.split_initialized_bytes(column_count, activation_bytes)?;
    if output.is_empty() {
        return Ok(());
    }
    let activations = prepare_activations(input, rows, k, activation_storage)?;
    let transpose = rows > 1 && n > 1;
    let columns = if transpose {
        &mut *column_major
    } else {
        &mut *output
    };
    rayon_decoded_columns(columns, rows, &|first, columns| {
        for (local, values) in columns.chunks_exact_mut(rows).enumerate() {
            let feature = first + local;
            for (row, value) in values.iter_mut().enumerate() {
                let mut sum = 0.;
                for block in 0..blocks {
                    let weight = &packed[feature * stride + block * format.bytes()
                        ..feature * stride + (block + 1) * format.bytes()];
                    let activation = &activations[(row * blocks + block) * ACTIVATION_BYTES
                        ..(row * blocks + block + 1) * ACTIVATION_BYTES];
                    let scale = half::f16::from_le_bytes([weight[0], weight[1]]).to_f32();
                    if !scale.is_finite() {
                        return Err(error(format!(
                            "q8_0 projection weight output {feature} block {block} has non-finite scale; select activation_arithmetic='float32' with block_layout_version=1"
                        )));
                    }
                    let activation_scale =
                        half::f16::from_le_bytes([activation[0], activation[1]]).to_f32();
                    sum += format.dot(weight, activation) as f32 * (scale * activation_scale);
                }
                *value = sum + bias.map_or(0., |bias| bias[feature]);
            }
        }
        Ok(())
    })?;
    if transpose {
        for (row, target) in output.chunks_exact_mut(n).enumerate() {
            for (feature, value) in target.iter_mut().enumerate() {
                *value = column_major[feature * rows + row];
            }
        }
    }
    onnx_runtime_ep_api::record_kernel_variant!(
        "packed_projection.q8_0",
        "grouped Q8 activation integer dots: rows={rows}, K={k}, N={n}"
    );
    Ok(())
}

#[cfg(test)]
thread_local! { static ACTIVATION_CONVERSIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
mod tests;

#[cfg(test)]
mod loop_order_study;
