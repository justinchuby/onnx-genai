//! ONNX `ConvTranspose` using deterministic output-owned NVRTC work.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::error::{driver_err, not_implemented};
use crate::runtime::{CudaRuntime, cuptr};

const BLOCK: u32 = 256;
const SOURCE: &str = r#"
#ifdef CONV_TRANSPOSE_HALF
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

using U = unsigned long long;
using I = long long;

template <typename T, typename Acc>
struct Storage {
  __device__ static Acc load(const T* values, U index) { return values[index]; }
  __device__ static void store(T* values, U index, Acc value) { values[index] = value; }
};
#ifdef CONV_TRANSPOSE_HALF
template <> struct Storage<__half, float> {
  __device__ static float load(const __half* values, U i) { return __half2float(values[i]); }
  __device__ static void store(__half* values, U i, float v) { values[i] = __float2half_rn(v); }
};
template <> struct Storage<__nv_bfloat16, float> {
  __device__ static float load(const __nv_bfloat16* values, U i) { return __bfloat162float(values[i]); }
  __device__ static void store(__nv_bfloat16* values, U i, float v) { values[i] = __float2bfloat16_rn(v); }
};
#endif

#define CT_ARGS(T) \
    const T* x, const T* weights, const T* bias, T* output, \
    U output_elements, U input_channels, U output_channels, \
    U input_channels_per_group, U output_channels_per_group, \
    U id, U ih, U iw, U od, U oh, U ow, U kd, U kh, U kw, \
    U sd, U sh, U sw, U dd, U dh, U dw, I pd, I ph, I pw

template <typename T, typename Acc, bool Volumetric>
__device__ void conv_transpose(CT_ARGS(T)) {
  using S = Storage<T, Acc>;
  const U output_plane = oh * ow, input_plane = ih * iw, kernel_plane = kh * kw;
  const U output_spatial = Volumetric ? od * output_plane : output_plane;
  const U input_spatial = Volumetric ? id * input_plane : input_plane;
  const U kernel_spatial = Volumetric ? kd * kernel_plane : kernel_plane;
  for (unsigned long long output_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       output_index < output_elements;
       output_index += (unsigned long long)gridDim.x * blockDim.x) {
    const unsigned long long batch = output_index / (output_channels * output_spatial);
    const unsigned long long within_batch = output_index % (output_channels * output_spatial);
    const unsigned long long output_channel = within_batch / output_spatial;
    const unsigned long long output_flat = within_batch % output_spatial;
    const U output_z = Volumetric ? output_flat / output_plane : 0;
    const U output_y = (Volumetric ? output_flat % output_plane : output_flat) / ow;
    const U output_x = output_flat % ow;
    const unsigned long long group = output_channel / output_channels_per_group;
    const unsigned long long output_in_group = output_channel % output_channels_per_group;
    const unsigned long long input_begin = group * input_channels_per_group;
    const unsigned long long input_end = input_begin + input_channels_per_group;
    Acc accumulated = bias ? S::load(bias, output_channel) : Acc(0);

    // Keep the CPU reference's input-channel/input-position/kernel-position order.
    for (unsigned long long input_channel = input_begin;
         input_channel < input_end; ++input_channel) {
      for (unsigned long long input_flat = 0; input_flat < input_spatial; ++input_flat) {
        const U input_z = Volumetric ? input_flat / input_plane : 0;
        const U input_y = (Volumetric ? input_flat % input_plane : input_flat) / iw;
        const U input_x = input_flat % iw;
        const Acc input_value = S::load(
            x, (batch * input_channels + input_channel) * input_spatial + input_flat);
        for (unsigned long long kernel_flat = 0; kernel_flat < kernel_spatial; ++kernel_flat) {
          if (Volumetric && I(input_z * sd + (kernel_flat / kernel_plane) * dd) - pd != I(output_z))
            continue;
          const U kernel_y = (Volumetric ? kernel_flat % kernel_plane : kernel_flat) / kw;
          const U kernel_x = kernel_flat % kw;
          if (I(input_y * sh + kernel_y * dh) - ph != I(output_y) ||
              I(input_x * sw + kernel_x * dw) - pw != I(output_x)) continue;
          const unsigned long long weight_index =
              (input_channel * output_channels_per_group + output_in_group) * kernel_spatial
              + kernel_flat;
          accumulated += input_value * S::load(weights, weight_index);
        }
      }
    }
    S::store(output, output_index, accumulated);
  }
}

#define CT_ENTRY(name, T, Acc, volumetric) \
  extern "C" __global__ void name(CT_ARGS(T)) { \
    conv_transpose<T, Acc, volumetric>(x, weights, bias, output, output_elements, \
        input_channels, output_channels, input_channels_per_group, output_channels_per_group, \
        id, ih, iw, od, oh, ow, kd, kh, kw, sd, sh, sw, dd, dh, dw, pd, ph, pw); \
  }
#define CT_ENTRIES(name, T, Acc) \
  CT_ENTRY(name##_2d, T, Acc, false) \
  CT_ENTRY(name##_3d, T, Acc, true)

#ifdef CONV_TRANSPOSE_HALF
CT_ENTRIES(conv_transpose_f16, __half, float)
CT_ENTRIES(conv_transpose_bf16, __nv_bfloat16, float)
#else
CT_ENTRIES(conv_transpose_f32, float, float)
CT_ENTRIES(conv_transpose_f64, double, double)
#endif
"#;

pub struct ConvTransposeFactory {
    pub runtime: Arc<CudaRuntime>,
    pub since_version: u64,
}

pub(super) const BFLOAT16_SINCE: u64 = 22;

enum Padding {
    Explicit(Vec<i64>),
    Inferred {
        shape: Option<Vec<i64>>,
        upper: bool,
    },
}

struct Parameters {
    dilations: Vec<i64>,
    group: usize,
    kernel_shape: Option<Vec<i64>>,
    output_padding: Vec<i64>,
    padding: Padding,
    strides: Vec<i64>,
}

impl KernelFactory for ConvTransposeFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let parameters = Parameters::parse(node, metadata_rank(input_shapes)?)?;
        parameters.validate_kernel(&input_shapes[1][2..])?;
        Ok(Box::new(ConvTransposeKernel {
            runtime: self.runtime.clone(),
            parameters,
            since_version: self.since_version,
        }))
    }
}

fn invalid(message: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("cuda_ep ConvTranspose: {message}"))
}

fn metadata_rank<T>(shapes: &[Vec<T>]) -> Result<usize> {
    let [x, w, ..] = shapes else {
        return Err(invalid("requires shape metadata for both X and W"));
    };
    spatial_rank(x.len(), w.len())
}

fn spatial_rank(x: usize, w: usize) -> Result<usize> {
    if !(3..=5).contains(&x) || w != x {
        return Err(invalid(format!(
            "X and W must have matching ranks in 3..=5 (1-D to 3-D spatial input); got {x} and {w}"
        )));
    }
    Ok(x - 2)
}

impl Parameters {
    fn parse(node: &Node, rank: usize) -> Result<Self> {
        let strides = int_list(node, "strides", rank, 1)?.unwrap_or_else(|| vec![1; rank]);
        let dilations = int_list(node, "dilations", rank, 1)?.unwrap_or_else(|| vec![1; rank]);
        let output_padding =
            int_list(node, "output_padding", rank, 0)?.unwrap_or_else(|| vec![0; rank]);
        if output_padding
            .iter()
            .enumerate()
            .any(|(axis, &value)| value >= strides[axis] && value >= dilations[axis])
        {
            return Err(invalid(
                "output_padding must be smaller than stride or dilation on each spatial axis",
            ));
        }
        let group = match node.attr("group") {
            None => 1,
            Some(attr) => attr
                .as_int()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|&value| value != 0)
                .ok_or_else(|| invalid("group must be a positive integer"))?,
        };
        let auto_pad = match node.attr("auto_pad") {
            None => "NOTSET",
            Some(attr) => attr
                .as_str()
                .ok_or_else(|| invalid("auto_pad must be a string"))?,
        };
        let output_shape = int_list(node, "output_shape", rank, 0)?;
        let pads = int_list(node, "pads", 2 * rank, 0)?;
        if !matches!(
            auto_pad,
            "" | "NOTSET" | "VALID" | "SAME_UPPER" | "SAME_LOWER"
        ) {
            return Err(invalid(format!(
                "auto_pad {auto_pad:?} must be NOTSET, VALID, SAME_UPPER or SAME_LOWER"
            )));
        }
        let padding = if output_shape.is_some() || matches!(auto_pad, "SAME_UPPER" | "SAME_LOWER") {
            if output_shape.is_none() && pads.is_some() {
                return Err(invalid("omit pads when using SAME_UPPER or SAME_LOWER"));
            }
            Padding::Inferred {
                shape: output_shape,
                upper: auto_pad == "SAME_UPPER",
            }
        } else if auto_pad == "VALID" {
            Padding::Explicit(vec![0; 2 * rank])
        } else {
            Padding::Explicit(pads.unwrap_or_else(|| vec![0; 2 * rank]))
        };
        Ok(Self {
            dilations,
            group,
            kernel_shape: int_list(node, "kernel_shape", rank, 1)?,
            output_padding,
            padding,
            strides,
        })
    }

    fn validate_kernel(&self, shape: &[usize]) -> Result<()> {
        if shape.len() != self.strides.len()
            || shape.contains(&0)
            || self.kernel_shape.as_ref().is_some_and(|kernel| {
                kernel
                    .iter()
                    .zip(shape)
                    .any(|(&attr, &weight)| usize::try_from(attr) != Ok(weight))
            })
        {
            return Err(invalid(format!(
                "W spatial dimensions {shape:?} must be positive and match kernel_shape {:?}",
                self.kernel_shape
            )));
        }
        Ok(())
    }
}

fn int_list(node: &Node, name: &str, count: usize, minimum: i64) -> Result<Option<Vec<i64>>> {
    node.attr(name)
        .map(|attr| {
            attr.as_ints()
                .filter(|values| {
                    values.len() == count && values.iter().all(|&value| value >= minimum)
                })
                .map(<[i64]>::to_vec)
                .ok_or_else(|| {
                    invalid(format!("{name} must contain {count} integers >= {minimum}"))
                })
        })
        .transpose()
}

pub(super) fn validate_claim(node: &Node, shapes: &[onnx_runtime_ir::Shape]) -> Result<()> {
    Parameters::parse(node, metadata_rank(shapes)?)?;
    Ok(())
}

pub(super) fn storage(dtype: DataType, opset: u64) -> std::result::Result<Storage, String> {
    match dtype {
        DataType::Float32 => Ok(Storage::Float32),
        DataType::Float16 => Ok(Storage::Float16),
        DataType::BFloat16 if opset >= BFLOAT16_SINCE => Ok(Storage::BFloat16),
        DataType::Float64 => Ok(Storage::Float64),
        DataType::BFloat16 => Err(format!(
            "BF16 requires opset {BFLOAT16_SINCE} or later; got {opset}"
        )),
        other => Err(format!(
            "X dtype {other:?} is unsupported; use Float32, Float16, Float64 or opset-{BFLOAT16_SINCE}+ BFloat16"
        )),
    }
}

pub(super) enum Storage {
    Float32,
    Float16,
    BFloat16,
    Float64,
}

impl Storage {
    fn function(
        &self,
        runtime: &CudaRuntime,
        volumetric: bool,
    ) -> Result<cudarc::driver::CudaFunction> {
        let entry = match (self, volumetric) {
            (Self::Float32, false) => "conv_transpose_f32_2d",
            (Self::Float32, true) => "conv_transpose_f32_3d",
            (Self::Float16, false) => "conv_transpose_f16_2d",
            (Self::Float16, true) => "conv_transpose_f16_3d",
            (Self::BFloat16, false) => "conv_transpose_bf16_2d",
            (Self::BFloat16, true) => "conv_transpose_bf16_3d",
            (Self::Float64, false) => "conv_transpose_f64_2d",
            (Self::Float64, true) => "conv_transpose_f64_3d",
        };
        if matches!(self, Self::Float16 | Self::BFloat16) {
            runtime.require_nvrtc_half_headers("ConvTranspose")?;
            runtime.nvrtc_function(
                "conv_transpose_half_v2",
                &format!("#define CONV_TRANSPOSE_HALF\n{SOURCE}"),
                entry,
            )
        } else {
            runtime.nvrtc_function("conv_transpose_float_v2", SOURCE, entry)
        }
    }
}

struct ConvTransposeKernel {
    runtime: Arc<CudaRuntime>,
    parameters: Parameters,
    since_version: u64,
}

impl Parameters {
    fn output_geometry(&self, input: &[usize], kernel: &[usize]) -> Result<(Vec<usize>, Vec<i64>)> {
        self.validate_kernel(kernel)?;
        let rank = input.len();
        let mut output = Vec::with_capacity(rank);
        let mut pad_begin = Vec::with_capacity(rank);
        for axis in 0..rank {
            let overflow = || {
                invalid(format!(
                    "spatial axis {axis} exceeds signed 64-bit geometry; reduce shape, stride or dilation"
                ))
            };
            let x = i64::try_from(input[axis]).map_err(|_| overflow())?;
            let k = i64::try_from(kernel[axis]).map_err(|_| overflow())?;
            let stride = i128::from(self.strides[axis]);
            let last =
                stride * i128::from(x - 1) + i128::from(self.dilations[axis]) * i128::from(k - 1);
            let base = last + 1 + i128::from(self.output_padding[axis]);
            let (size, begin) = match &self.padding {
                Padding::Explicit(pads) => {
                    let begin = i128::from(pads[axis]);
                    (base - begin - i128::from(pads[axis + rank]), begin)
                }
                Padding::Inferred { shape, upper } => {
                    let desired = shape
                        .as_ref()
                        .map_or(i128::from(x) * stride, |s| i128::from(s[axis]));
                    let total = base - desired;
                    // floor corrects the odd split even when a stride leaves a hole
                    // (negative generated padding).
                    let small = total.div_euclid(2);
                    (desired, if *upper { small } else { total - small })
                }
            };
            if size < 0 {
                return Err(invalid(format!(
                    "pads produce negative output extent {size} on spatial axis {axis}; reduce pads or specify output_shape"
                )));
            }
            // The device translates signed coordinates only for existing X cells.
            if x > 0 && (i64::try_from(last).is_err() || i64::try_from(last - begin).is_err()) {
                return Err(overflow());
            }
            output.push(
                usize::try_from(i64::try_from(size).map_err(|_| overflow())?)
                    .map_err(|_| overflow())?,
            );
            pad_begin.push(i64::try_from(begin).map_err(|_| overflow())?);
        }
        Ok((output, pad_begin))
    }
}

impl Kernel for ConvTransposeKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(2..=3).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: expected X, W, optional B and one output".into(),
            ));
        }
        if inputs.iter().any(|input| !input.is_contiguous()) || !outputs[0].is_contiguous() {
            return Err(not_implemented("ConvTranspose with strided tensors"));
        }
        let x = &inputs[0];
        let weights = &inputs[1];
        let rank = spatial_rank(x.shape.len(), weights.shape.len())?;
        if x.dtype != weights.dtype || outputs[0].dtype != x.dtype {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: X, W, and Y dtypes must match".into(),
            ));
        }
        let storage = storage(x.dtype, self.since_version).map_err(invalid)?;
        let group = self.parameters.group;
        if weights.shape[0] != x.shape[1]
            || !x.shape[1].is_multiple_of(group)
            || weights.shape[1] == 0
        {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: invalid grouped channel geometry".into(),
            ));
        }
        let output_channels = weights.shape[1]
            .checked_mul(group)
            .ok_or_else(|| EpError::KernelFailed("output channel overflow".into()))?;
        let (output_spatial, pad_begin) = self
            .parameters
            .output_geometry(&x.shape[2..], &weights.shape[2..])?;
        let expected = [x.shape[0], output_channels]
            .into_iter()
            .chain(output_spatial.iter().copied())
            .collect::<Vec<_>>();
        if outputs[0].shape != expected {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep ConvTranspose: output shape {:?}, expected {expected:?}",
                outputs[0].shape
            )));
        }
        let bias = inputs.get(2);
        if bias.is_some_and(|value| value.dtype != x.dtype || value.shape != [output_channels]) {
            return Err(EpError::KernelFailed(
                "cuda_ep ConvTranspose: bias must match output channels and dtype".into(),
            ));
        }
        if outputs[0].numel() == 0 {
            return Ok(());
        }
        let function = storage.function(&self.runtime, rank == 3)?;
        let x_pointer = cuptr(x.data_ptr::<u8>() as *const c_void);
        let weights_pointer = cuptr(weights.data_ptr::<u8>() as *const c_void);
        let bias_pointer = bias
            .map(|value| cuptr(value.data_ptr::<u8>() as *const c_void))
            .unwrap_or(0);
        let output_pointer = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);
        let output_elements = outputs[0].numel() as u64;
        let with_leading_ones = |values: &[usize]| {
            let mut axes = [1; 3];
            for (dst, &src) in axes[3 - rank..].iter_mut().zip(values) {
                *dst = src as u64;
            }
            axes
        };
        let unsigned_axes = |values: &[i64]| {
            let mut axes = [1; 3];
            for (dst, &src) in axes[3 - rank..].iter_mut().zip(values) {
                *dst = src as u64;
            }
            axes
        };
        let channels = [
            x.shape[1],
            output_channels,
            x.shape[1] / group,
            weights.shape[1],
        ]
        .map(|value| value as u64);
        let dimensions = [
            with_leading_ones(&x.shape[2..]),
            with_leading_ones(&output_spatial),
            with_leading_ones(&weights.shape[2..]),
            unsigned_axes(&self.parameters.strides),
            unsigned_axes(&self.parameters.dilations),
        ];
        let mut pads = [0; 3];
        pads[3 - rank..].copy_from_slice(&pad_begin);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x_pointer)
            .arg(&weights_pointer)
            .arg(&bias_pointer)
            .arg(&output_pointer)
            .arg(&output_elements);
        for dimension in channels.iter().chain(dimensions.iter().flatten()) {
            builder.arg(dimension);
        }
        for pad in &pads {
            builder.arg(pad);
        }
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (
                    output_elements.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32,
                    1,
                    1,
                ),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch ConvTranspose", error))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::standard_claims::unsupported_reason_at_opset;
    use onnx_runtime_ir::{NodeId, ValueId, static_shape};

    fn node(attributes: &[(&str, Attribute)]) -> Node {
        let mut node = Node::new(
            NodeId(0),
            "ConvTranspose",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(2)],
        );
        node.attributes.extend(
            attributes
                .iter()
                .map(|(name, attr)| ((*name).into(), attr.clone())),
        );
        node
    }

    #[test]
    fn conv_transpose_claim_observes_dtype_and_rank_boundaries() {
        let op = node(&[]);
        let shapes = [static_shape([1, 1, 2, 2, 2]), static_shape([1, 1, 3, 2, 1])];
        let claim = |opset, dtype| unsupported_reason_at_opset(&op, opset, &shapes, &[dtype; 2]);
        assert!(
            claim(21, DataType::BFloat16)
                .unwrap()
                .contains("requires opset 22")
        );
        assert_eq!(claim(22, DataType::BFloat16), None);
        assert_eq!(claim(11, DataType::Float64), None);
        assert_eq!(claim(22, DataType::Float64), None);

        for invalid_shapes in [
            [static_shape([1; 6]), static_shape([1; 6])],
            [static_shape([1; 5]), static_shape([1; 4])],
        ] {
            let reason =
                unsupported_reason_at_opset(&op, 22, &invalid_shapes, &[DataType::Float32; 2])
                    .expect("CUDA must not claim unsupported or mismatched ranks");
            assert!(reason.contains("matching ranks in 3..=5"), "{reason}");
        }
    }

    #[test]
    fn conv_transpose_rejects_malformed_padding_at_claim() {
        let shapes = [static_shape([1, 1, 2]), static_shape([1, 1, 3])];
        for (attr, value) in [
            ("output_shape", Attribute::Ints(vec![3, 4])),
            ("output_shape", Attribute::Ints(vec![-1])),
            ("output_padding", Attribute::Ints(vec![1])),
            ("auto_pad", Attribute::String(b"SAME".to_vec())),
        ] {
            let reason = unsupported_reason_at_opset(
                &node(&[(attr, value)]),
                22,
                &shapes,
                &[DataType::Float32; 2],
            )
            .expect("malformed ConvTranspose attributes must not be claimed");
            assert!(reason.contains(attr), "{reason}");
        }
    }

    #[test]
    fn conv_transpose_geometry_checks_empty_input_and_coordinate_overflow() {
        let op = node(&[("strides", Attribute::Ints(vec![2]))]);
        let parameters = Parameters::parse(&op, 1).unwrap();
        assert_eq!(
            parameters.output_geometry(&[0], &[3]).unwrap(),
            (vec![1], vec![0])
        );

        let op = node(&[("dilations", Attribute::Ints(vec![i64::MAX]))]);
        let reason = Parameters::parse(&op, 1)
            .unwrap()
            .output_geometry(&[2], &[2])
            .unwrap_err()
            .to_string();
        assert!(
            reason.contains("exceeds signed 64-bit geometry"),
            "{reason}"
        );
    }
}
