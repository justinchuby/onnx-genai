//! Concrete tensor contracts shared by the standalone packing and routing kernels.

use crate::{EpError, Result, TensorMut, TensorView};
use onnx_runtime_ir::{
    DataType, Node, Shape,
    dequantize_packed::Metadata,
    packed_routed::{OPS, Op, Packing},
};

pub fn unsupported_reason(node: &Node, _: &[Shape], dtypes: &[DataType]) -> Option<String> {
    if matches!(node.domain.as_str(), "" | "ai.onnx") && OPS.contains(&node.op_type.as_str()) {
        Op::from_node(node)
            .and_then(|op| op.validate_dtypes(dtypes))
            .err()
    } else {
        None
    }
}

pub fn fail(op: Op, detail: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{}: {detail}", op.name()))
}

fn elements(op: Op, shape: &[usize]) -> Result<usize> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |n, &d| n.checked_mul(d))
        .ok_or_else(|| {
            fail(
                op,
                format!("shape {shape:?} exceeds the addressable element count"),
            )
        })
}

pub fn validate(op: Op, x: &[TensorView], y: &[TensorMut]) -> Result<()> {
    let valid_count = match op {
        Op::Pack(..) => x.len() == 1,
        Op::Unpack(..) => x.len() == 2,
        Op::GroupedMatMul { .. } => (3..=4).contains(&x.len()),
    };
    if !valid_count || y.len() != 1 {
        return Err(fail(
            op,
            format!(
                "invalid operand count ({} inputs, {} outputs); check the operator signature",
                x.len(),
                y.len()
            ),
        ));
    }
    let dtypes: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if i == 3 && t.is_absent() {
                DataType::Undefined
            } else {
                t.dtype
            }
        })
        .collect();
    op.validate_dtypes(&dtypes).map_err(EpError::KernelFailed)?;
    let required = if matches!(op, Op::GroupedMatMul { .. }) {
        3
    } else {
        x.len()
    };
    if dtypes[..required].contains(&DataType::Undefined)
        || x[..required]
            .iter()
            .any(|t| !t.shape.contains(&0) && t.is_absent())
    {
        return Err(fail(
            op,
            "missing a required input tensor; supply all non-optional operands",
        ));
    }
    if matches!(op, Op::Unpack(..)) && !x[1].shape.is_empty() {
        return Err(fail(
            op,
            format!("count must be a scalar INT64, got shape {:?}", x[1].shape),
        ));
    }
    let wanted = match op {
        Op::Pack(_, dt) => dt,
        Op::Unpack(_) => DataType::Uint8,
        Op::GroupedMatMul { .. } => x[0].dtype,
    };
    if wanted != y[0].dtype {
        return Err(fail(
            op,
            format!("output must have dtype {wanted:?}, got {:?}", y[0].dtype),
        ));
    }
    for (shape, dtype) in x
        .iter()
        .map(|t| (t.shape, t.dtype))
        .chain([(y[0].shape, y[0].dtype)])
    {
        if dtype != DataType::Undefined {
            elements(op, shape)?
                .checked_mul(dtype.byte_size())
                .ok_or_else(|| {
                    fail(
                        op,
                        format!("tensor {dtype:?} {shape:?} exceeds the addressable byte count"),
                    )
                })?;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub struct PackLayout {
    pub packing: Packing,
    pub count: usize,
    pub words: usize,
    pub word_bits: usize,
    pub inner: usize,
    pub lanes: usize,
}

pub fn pack_layout(
    op: Op,
    x: &TensorView,
    y: &TensorMut,
    count: Option<i64>,
) -> Result<PackLayout> {
    let (packing, codes, packed, word_bits) = match op {
        Op::Pack(packing, dtype) => (packing, x.shape, y.shape, dtype.byte_size() * 8),
        Op::Unpack(packing) => (packing, y.shape, x.shape, x.dtype.byte_size() * 8),
        Op::GroupedMatMul { .. } => unreachable!(),
    };
    let axis = packing.axis(x.shape.len()).map_err(|why| fail(op, why))?;
    if codes.len() != packed.len()
        || codes
            .iter()
            .zip(packed)
            .enumerate()
            .any(|(i, (a, b))| i != axis && a != b)
    {
        return Err(fail(
            op,
            format!(
                "input {:?} and output {:?} must have matching rank and non-axis dimensions (axis {axis})",
                x.shape, y.shape
            ),
        ));
    }
    let n = codes[axis];
    if let Some(count) = count
        && usize::try_from(count) != Ok(n)
    {
        return Err(fail(
            op,
            format!(
                "count {count} must be nonnegative and equal the output axis {axis} extent {n}"
            ),
        ));
    }
    let words = packing.words(n, word_bits).map_err(|why| fail(op, why))?;
    if words != packed[axis] {
        return Err(fail(
            op,
            format!(
                "packed axis {axis} must contain exactly {words} words for {n} codes, got {}",
                packed[axis]
            ),
        ));
    }
    let total = elements(op, y.shape)?;
    let inner = if total == 0 {
        0
    } else {
        codes[axis + 1..].iter().product()
    };
    Ok(PackLayout {
        packing,
        count: n,
        words,
        word_bits,
        inner,
        lanes: total,
    })
}

#[derive(Clone, Copy)]
pub struct GroupLayout {
    pub k: usize,
    pub n: usize,
    pub groups: usize,
    pub selections: usize,
    pub routes: usize,
    pub lanes: usize,
    pub transposed: bool,
    pub bias: bool,
}

pub fn group_layout(op: Op, x: &[TensorView], y: &TensorMut) -> Result<GroupLayout> {
    let bias = x
        .get(3)
        .filter(|b| b.dtype != DataType::Undefined && !b.is_absent())
        .map(|b| b.shape);
    group_layout_shapes(op, x[0].shape, x[1].shape, x[2].shape, bias, y.shape)
}

pub fn packed_group_layout(
    metadata: &Metadata,
    x: &[TensorView],
    y: &[TensorMut],
) -> Result<GroupLayout> {
    let fail = |detail: &str| EpError::KernelFailed(format!("PackedGroupedMatMul: {detail}"));
    let [a, packed, idx] = x else {
        return Err(fail(
            "expected activation, packed weights, and group indices",
        ));
    };
    let [out] = y else {
        return Err(fail("expected one output"));
    };
    let [groups, n, k] = metadata.shape() else {
        return Err(fail("decoded weight shape must be [groups,N,K]"));
    };
    if metadata.dtype() != DataType::Float32
        || a.dtype != DataType::Float32
        || out.dtype != DataType::Float32
        || !matches!(idx.dtype, DataType::Int32 | DataType::Int64)
    {
        return Err(fail(
            "activation, decoded weights, and output must be FLOAT, with INT32 or INT64 group indices",
        ));
    }
    metadata
        .validate_input(packed.dtype, packed.shape)
        .map_err(|detail| fail(&detail))?;
    for tensor in x {
        tensor.validate()?;
    }
    out.validate()?;
    group_layout_shapes(
        Op::GroupedMatMul { transposed: true },
        a.shape,
        &[*groups, *n, *k],
        idx.shape,
        None,
        out.shape,
    )
}

fn group_layout_shapes(
    op: Op,
    a: &[usize],
    w: &[usize],
    idx: &[usize],
    bias: Option<&[usize]>,
    out: &[usize],
) -> Result<GroupLayout> {
    let Op::GroupedMatMul { transposed } = op else {
        unreachable!()
    };
    let ([m, k], [groups, w1, w2], [rows, selections]) = (a, w, idx) else {
        return Err(fail(
            op,
            format!(
                "expected input [M,K], weights rank 3 and indices [M,S], got {a:?}, {w:?}, {idx:?}"
            ),
        ));
    };
    let (&wk, &n) = if transposed { (w2, w1) } else { (w1, w2) };
    if m != rows || *k != wk {
        return Err(fail(
            op,
            format!(
                "inconsistent token or contraction dimensions in {a:?}, {w:?}, {idx:?} (transB={transposed})"
            ),
        ));
    }
    if let Some(b) = bias
        && b != [*groups, n]
    {
        return Err(fail(
            op,
            format!("bias must have shape [{groups}, {n}], got {b:?}"),
        ));
    }
    let expected = [*m, *selections, n];
    if out != expected {
        return Err(fail(
            op,
            format!("output must have shape {expected:?}, got {out:?}"),
        ));
    }
    Ok(GroupLayout {
        k: *k,
        n,
        groups: *groups,
        selections: *selections,
        routes: elements(op, idx)?,
        lanes: elements(op, &expected)?,
        transposed,
        bias: bias.is_some(),
    })
}

#[derive(Clone, Copy)]
pub struct AffineGrouped {
    pub depth: usize,
    pub block: usize,
    pub zero: u8,
}

impl AffineGrouped {
    pub fn from_node(node: &Node) -> Result<Self> {
        let attr = |name| {
            node.attr(name)
                .and_then(|a| a.as_int())
                .and_then(|i| usize::try_from(i).ok())
                .ok_or_else(|| {
                    EpError::KernelFailed(format!(
                        "{} requires a nonnegative integer {name}",
                        node.op_type
                    ))
                })
        };
        let depth = attr("depth")?;
        let block = attr("block_size")?;
        let zero = attr("zero")?;
        if block == 0 || depth == 0 || !depth.is_multiple_of(block) || zero > 15 {
            return Err(EpError::KernelFailed(format!(
                "{} requires positive depth divisible by block_size and a four-bit zero point",
                node.op_type
            )));
        }
        Ok(Self {
            depth,
            block,
            zero: zero as u8,
        })
    }

    pub fn layout(self, x: &[TensorView], y: &[TensorMut]) -> Result<GroupLayout> {
        let fail =
            |detail: &str| EpError::KernelFailed(format!("AffineQuantizedGroupedMatMul: {detail}"));
        let [a, packed, idx, scale, ..] = x else {
            return Err(fail("expected four or five inputs"));
        };
        let [out] = y else {
            return Err(fail("expected one output"));
        };
        let [e, n, words] = packed.shape else {
            return Err(fail("expected rank-three packed weight bank"));
        };
        let blocks = self.depth / self.block;
        if x.len() > 5
            || a.dtype != DataType::Float32
            || out.dtype != DataType::Float32
            || !matches!(idx.dtype, DataType::Int32 | DataType::Int64)
            || packed.dtype != DataType::Uint8
            || scale.dtype != DataType::Float32
            || *words != self.depth.div_ceil(2)
            || scale.shape != [*e, *n, blocks]
            || x.get(4).is_some_and(|z| {
                z.dtype != DataType::Uint8 || z.shape != [*e, *n, blocks.div_ceil(2)]
            })
        {
            return Err(fail(
                "packed weights, scales or zero points do not match the expert-bank layout; check depth and block_size",
            ));
        }
        let op = Op::GroupedMatMul { transposed: true };
        elements(op, &[*e, *n, self.depth])?;
        for t in x {
            t.validate()?;
        }
        out.validate()?;
        group_layout_shapes(
            op,
            a.shape,
            &[*e, *n, self.depth],
            idx.shape,
            None,
            out.shape,
        )
    }
}
