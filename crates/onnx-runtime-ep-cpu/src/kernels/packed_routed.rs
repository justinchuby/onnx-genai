use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};
use onnx_runtime_ep_api::packed_routed::{
    GroupLayout, PackLayout, fail, group_layout, pack_layout, validate,
};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
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

fn pack(op: Op, l: PackLayout, x: &[u8]) -> Result<Vec<u32>> {
    let mut y = vec![0; l.lanes];
    if l.lanes == 0 {
        return Ok(y);
    }
    let p = l.packing;
    for row in 0..l.lanes / l.words {
        let outer = row / l.inner;
        let inner = row % l.inner;
        for (g, start) in (0..l.count).step_by(p.group).enumerate() {
            let mut index = 0u64;
            for col in (start..start + p.group.min(l.count - start)).rev() {
                let i = (outer * l.count + col) * l.inner + inner;
                let digit = u64::from(x[i]);
                if digit >= p.radix {
                    return Err(fail(
                        op,
                        format!(
                            "code X[{i}] is {digit}; codes must be less than radix {}",
                            p.radix
                        ),
                    ));
                }
                index = index * p.radix + digit;
            }
            let mut bit = g * p.bits;
            let mut remaining = p.bits;
            while remaining != 0 {
                let offset = bit % l.word_bits;
                let take = remaining.min(l.word_bits - offset);
                let i = (outer * l.words + bit / l.word_bits) * l.inner + inner;
                y[i] |= ((index & ((1u64 << take) - 1)) << offset) as u32;
                index >>= take;
                remaining -= take;
                bit += take;
            }
        }
    }
    Ok(y)
}

fn unpack<W: Copy + Into<u64>>(op: Op, l: PackLayout, x: &[W]) -> Result<Vec<u8>> {
    let mut y = vec![0; l.lanes];
    if l.lanes == 0 {
        return Ok(y);
    }
    let p = l.packing;
    for row in 0..l.lanes / l.count {
        let outer = row / l.inner;
        let inner = row % l.inner;
        for (g, start) in (0..l.count).step_by(p.group).enumerate() {
            let mut bit = g * p.bits;
            let mut index = 0u64;
            let mut used = 0;
            while used < p.bits {
                let offset = bit % l.word_bits;
                let take = (p.bits - used).min(l.word_bits - offset);
                let i = (outer * l.words + bit / l.word_bits) * l.inner + inner;
                index |= ((x[i].into() >> offset) & ((1u64 << take) - 1)) << used;
                used += take;
                bit += take;
            }
            if index >= p.limit {
                return Err(fail(
                    op,
                    format!(
                        "row {row} has invalid packed group {g}: index {index} must be less than {}",
                        p.limit
                    ),
                ));
            }
            for col in start..start + p.group.min(l.count - start) {
                y[(outer * l.count + col) * l.inner + inner] = (index % p.radix) as u8;
                index /= p.radix;
            }
        }
    }
    Ok(y)
}

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
    w: &[F],
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
                sum = sum + x[token * l.k + k] * w[g * l.k * l.n + i];
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
        match op {
            Op::Pack(_, dt) => {
                let l = pack_layout(op, &x[0], &y[0], None)?;
                let data = pack(op, l, &to_dense::<u8>(&x[0])?)?;
                if dt == DataType::Uint32 {
                    write_dense(&mut y[0], &data)
                } else {
                    write_dense(
                        &mut y[0],
                        &data.into_iter().map(|n| n as u8).collect::<Vec<_>>(),
                    )
                }
            }
            Op::Unpack(_) => {
                let count = to_dense::<i64>(&x[1])?[0];
                let l = pack_layout(op, &x[0], &y[0], Some(count))?;
                let data = if x[0].dtype == DataType::Uint32 {
                    unpack(op, l, &to_dense::<u32>(&x[0])?)?
                } else {
                    unpack(op, l, &to_dense::<u8>(&x[0])?)?
                };
                write_dense(&mut y[0], &data)
            }
            Op::GroupedMatMul { .. } => {
                let l = group_layout(op, x, &y[0])?;
                let groups = route_indices(op, &x[2], l.groups)?;
                if l.lanes == 0 {
                    return Ok(());
                }
                let b = l.bias.then(|| &x[3]);
                if x[0].dtype == DataType::Float64 {
                    let bias = b.map(to_dense::<f64>).transpose()?;
                    let data = grouped(
                        l,
                        &to_dense::<f64>(&x[0])?,
                        &to_dense::<f64>(&x[1])?,
                        &groups,
                        bias.as_deref(),
                        |v| v,
                    );
                    write_dense(&mut y[0], &data)
                } else {
                    let widen = |t| to_dense_f32_widen(op.name(), t);
                    let bias = b.map(widen).transpose()?;
                    let data = grouped(
                        l,
                        &widen(&x[0])?,
                        &widen(&x[1])?,
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
    }
}
