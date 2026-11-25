use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};
use onnx_runtime_ep_api::packed_routed::{
    GroupLayout, fail, group_layout, packed_group_layout, validate,
};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::dequantize_packed::Metadata;
use onnx_runtime_ir::{DataType, Node, packed_routed::Op};

pub(super) struct Factory;

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Native(
            Op::from_node(node).map_err(EpError::KernelFailed)?,
        )))
    }
}

struct Native(Op);

fn route_indices(op: Op, idx: &TensorView, groups: usize) -> Result<Vec<usize>> {
    let convert = |i, value| {
        usize::try_from(value)
            .ok()
            .filter(|&g| g < groups)
            .ok_or_else(|| {
                fail(
                    op,
                    format!("group_indices[{i}] is {value}; select a group in [0, {groups})"),
                )
            })
    };
    if idx.dtype == DataType::Int32 {
        to_dense::<i32>(idx)?
            .into_iter()
            .enumerate()
            .map(|(i, v)| convert(i, i64::from(v)))
            .collect()
    } else {
        to_dense::<i64>(idx)?
            .into_iter()
            .enumerate()
            .map(|(i, v)| convert(i, v))
            .collect()
    }
}

fn grouped<F>(
    l: GroupLayout,
    x: &[F],
    weight: impl Fn(usize) -> F,
    groups: &[usize],
    b: Option<&[F]>,
    round: impl Fn(F) -> F,
) -> Vec<F>
where
    F: Copy + Default + std::ops::Add<Output = F> + std::ops::Mul<Output = F>,
{
    let mut y = vec![F::default(); l.lanes];
    for (route, row) in y.chunks_exact_mut(l.n).enumerate() {
        let g = groups[route];
        let token = route / l.selections;
        for (n, out) in row.iter_mut().enumerate() {
            let mut sum = F::default();
            for k in 0..l.k {
                let i = if l.transposed {
                    n * l.k + k
                } else {
                    k * l.n + n
                };
                sum = sum + x[token * l.k + k] * weight(g * l.k * l.n + i);
            }
            // The ONNX function applies Add after the MatMul result is stored.
            *out = if let Some(b) = b {
                round(sum) + b[g * l.n + n]
            } else {
                sum
            };
        }
    }
    y
}

impl Kernel for Native {
    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        let op = self.0;
        validate(op, x, y)?;
        let l = group_layout(op, x, &y[0])?;
        let groups = route_indices(op, &x[2], l.groups)?;
        if l.lanes == 0 {
            return Ok(());
        }
        let b = l.bias.then(|| &x[3]);
        if x[0].dtype == DataType::Float64 {
            let bias = b.map(to_dense::<f64>).transpose()?;
            let weights = to_dense::<f64>(&x[1])?;
            let data = grouped(
                l,
                &to_dense::<f64>(&x[0])?,
                |i| weights[i],
                &groups,
                bias.as_deref(),
                |v| v,
            );
            write_dense(&mut y[0], &data)
        } else {
            let widen = |t| to_dense_f32_widen(op.name(), t);
            let bias = b.map(widen).transpose()?;
            let weights = widen(&x[1])?;
            let data = grouped(
                l,
                &widen(&x[0])?,
                |i| weights[i],
                &groups,
                bias.as_deref(),
                |v| match x[0].dtype {
                    DataType::Float16 => half::f16::from_f32(v).to_f32(),
                    DataType::BFloat16 => half::bf16::from_f32(v).to_f32(),
                    _ => v,
                },
            );
            write_dense_f32_narrow(op.name(), &mut y[0], &data)
        }
    }
}

pub(super) struct AffineFactory;

impl KernelFactory for AffineFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(
            onnx_runtime_ep_api::packed_routed::AffineGrouped::from_node(node).map(Affine)?,
        ))
    }
}

struct Affine(onnx_runtime_ep_api::packed_routed::AffineGrouped);

pub(super) struct AffineBank<'a> {
    codes: std::borrow::Cow<'a, [u8]>,
    scales: Vec<f32>,
    zeros: Option<std::borrow::Cow<'a, [u8]>>,
    depth: usize,
    block: usize,
    zero: u8,
}

impl<'a> AffineBank<'a> {
    pub(super) fn new(
        codes: &'a TensorView<'_>,
        scales: &TensorView<'_>,
        zeros: Option<&'a TensorView<'_>>,
        [experts, width, depth]: [usize; 3],
        block: usize,
        zero: u8,
    ) -> Result<Self> {
        let op = Op::GroupedMatMul { transposed: true };
        if block == 0 || !depth.is_multiple_of(block) || zero > 15 {
            return Err(fail(
                op,
                "affine block must divide projection depth; zero point must fit UINT4",
            ));
        }
        let blocks = depth / block;
        if codes.dtype != DataType::Uint8
            || codes.shape != [experts, width, depth.div_ceil(2)]
            || scales.dtype != DataType::Float32
            || scales.shape != [experts, width, blocks]
            || zeros.is_some_and(|z| {
                z.dtype != DataType::Uint8 || z.shape != [experts, width, blocks.div_ceil(2)]
            })
        {
            return Err(fail(
                op,
                format!(
                    "inconsistent packed expert bank for [experts, width, depth]=[{experts}, {width}, {depth}], block={block}"
                ),
            ));
        }
        let scales = super::to_dense_f32(scales)?;
        if scales.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
            return Err(fail(
                op,
                "DequantizeLinear scales must be positive and finite",
            ));
        }
        fn read<'a>(view: &'a TensorView<'_>) -> Result<std::borrow::Cow<'a, [u8]>> {
            if !view.is_contiguous() {
                return super::to_dense_bytes(view).map(std::borrow::Cow::Owned);
            }
            // SAFETY: the geometry above validated a one-byte tensor.
            Ok(std::borrow::Cow::Borrowed(unsafe {
                std::slice::from_raw_parts(view.data_ptr::<u8>(), view.numel())
            }))
        }
        Ok(Self {
            codes: read(codes)?,
            scales,
            zeros: zeros.map(read).transpose()?,
            depth,
            block,
            zero,
        })
    }

    fn weight(&self, row: usize, col: usize) -> f32 {
        let blocks = self.depth / self.block;
        let nibble = |bytes: &[u8], col: usize, width: usize| {
            (bytes[row * width.div_ceil(2) + col / 2] >> ((col % 2) * 4)) & 15
        };
        let p = col / self.block;
        let z = self
            .zeros
            .as_ref()
            .map_or(self.zero, |z| nibble(z, p, blocks));
        f32::from(i16::from(nibble(&self.codes, col, self.depth)) - i16::from(z))
            * self.scales[row * blocks + p]
    }

    pub(super) fn dot(&self, row: usize, input: &[f32]) -> f32 {
        input
            .iter()
            .enumerate()
            .fold(0.0, |sum, (col, &x)| sum + x * self.weight(row, col))
    }
}

pub(super) struct PackedFactory;

impl KernelFactory for PackedFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Packed {
            metadata: Metadata::from_node(node).map_err(|detail| {
                EpError::KernelFailed(format!("PackedGroupedMatMul: {detail}"))
            })?,
        }))
    }
}

struct Packed {
    metadata: Metadata,
}

impl Kernel for Packed {
    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        let l = packed_group_layout(&self.metadata, x, y)?;
        if l.lanes == 0 {
            return Ok(());
        }
        if x.iter().any(|tensor| !tensor.is_contiguous()) || !y[0].is_contiguous() {
            return Err(EpError::KernelFailed(
                "PackedGroupedMatMul: inputs and output must use contiguous storage".into(),
            ));
        }
        let routes = route_indices(Op::GroupedMatMul { transposed: true }, &x[2], l.groups)?;
        let activation = super::to_dense_f32(&x[0])?;
        // SAFETY: packed_group_layout validated the one-dimensional UINT8
        // extent, and the contiguous CPU tensor remains live for this call.
        let packed = unsafe {
            std::slice::from_raw_parts(x[1].data_ptr::<u8>(), self.metadata.packed_bytes())
        };
        let (width, bytes) = self.metadata.format().block();
        let blocks = l.k / width;
        let decode = super::dequantize_packed::decoder(self.metadata.format());
        let mut values = [0.0f32; 256];
        let mut output = vec![0.0f32; l.lanes];
        for (route, row) in output.chunks_exact_mut(l.n).enumerate() {
            let expert = routes[route];
            let token = route / l.selections;
            for (column, out) in row.iter_mut().enumerate() {
                let packed_row = (expert * l.n + column) * blocks * bytes;
                let mut sum = 0.0f32;
                for block in 0..blocks {
                    let source = packed_row + block * bytes;
                    decode(&packed[source..source + bytes], &mut values[..width]);
                    let activation =
                        &activation[token * l.k + block * width..token * l.k + (block + 1) * width];
                    for (&a, &weight) in activation.iter().zip(&values[..width]) {
                        sum += a * weight;
                    }
                }
                *out = sum;
            }
        }
        write_dense(&mut y[0], &output)
    }
}

impl Kernel for Affine {
    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        let l = self.0.layout(x, y)?;
        let op = Op::GroupedMatMul { transposed: true };
        let routes = route_indices(op, &x[2], l.groups)?;
        let bank = AffineBank::new(
            &x[1],
            &x[3],
            x.get(4),
            [l.groups, l.n, l.k],
            self.0.block,
            self.0.zero,
        )?;
        if l.lanes == 0 {
            return Ok(());
        }
        let data = grouped(
            l,
            &super::to_dense_f32(&x[0])?,
            |i| bank.weight(i / l.k, i % l.k),
            &routes,
            None,
            |v| v,
        );
        write_dense(&mut y[0], &data)
    }
}
