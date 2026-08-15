//! Deterministic fixed-rank-order reduction kernels for the in-process oracle.

use half::{bf16, f16};

use crate::{CommError, CommResult, DType, ReduceOp};

pub(crate) fn reduce_buffers(
    inputs: &[Vec<u8>],
    dtype: DType,
    op: ReduceOp,
) -> CommResult<Vec<u8>> {
    let Some(first) = inputs.first() else {
        return Err(CommError::InvalidArgument(
            "reduction requires at least one rank".into(),
        ));
    };
    if inputs.iter().any(|input| input.len() != first.len()) {
        return Err(CommError::InvalidArgument(
            "reduction inputs have different byte lengths".into(),
        ));
    }
    if !first.len().is_multiple_of(dtype.size()) {
        return Err(CommError::InvalidArgument(
            "reduction extent is not dtype-aligned".into(),
        ));
    }
    match dtype {
        DType::F32 => reduce_f32(inputs, op),
        DType::F16 => reduce_f16(inputs, op),
        DType::BF16 => reduce_bf16(inputs, op),
        DType::I64 => reduce_i64(inputs, op),
        DType::I32 => reduce_i32(inputs, op),
        DType::U8 => reduce_u8(inputs, op),
    }
}

fn float_combine(lhs: f32, rhs: f32, op: ReduceOp) -> f32 {
    match op {
        ReduceOp::Sum => lhs + rhs,
        ReduceOp::Product => lhs * rhs,
        ReduceOp::Min => {
            if lhs.total_cmp(&rhs).is_le() {
                lhs
            } else {
                rhs
            }
        }
        ReduceOp::Max => {
            if lhs.total_cmp(&rhs).is_ge() {
                lhs
            } else {
                rhs
            }
        }
    }
}

fn reduce_f32(inputs: &[Vec<u8>], op: ReduceOp) -> CommResult<Vec<u8>> {
    let mut output = inputs[0].clone();
    for input in &inputs[1..] {
        for (out, rhs) in output.chunks_exact_mut(4).zip(input.chunks_exact(4)) {
            let lhs = f32::from_le_bytes(out.try_into().expect("four-byte chunk"));
            let rhs = f32::from_le_bytes(rhs.try_into().expect("four-byte chunk"));
            out.copy_from_slice(&float_combine(lhs, rhs, op).to_le_bytes());
        }
    }
    Ok(output)
}

fn reduce_f16(inputs: &[Vec<u8>], op: ReduceOp) -> CommResult<Vec<u8>> {
    let mut output = inputs[0].clone();
    for input in &inputs[1..] {
        for (out, rhs) in output.chunks_exact_mut(2).zip(input.chunks_exact(2)) {
            let lhs = f16::from_le_bytes(out.try_into().expect("two-byte chunk")).to_f32();
            let rhs = f16::from_le_bytes(rhs.try_into().expect("two-byte chunk")).to_f32();
            out.copy_from_slice(&f16::from_f32(float_combine(lhs, rhs, op)).to_le_bytes());
        }
    }
    Ok(output)
}

fn reduce_bf16(inputs: &[Vec<u8>], op: ReduceOp) -> CommResult<Vec<u8>> {
    let mut output = inputs[0].clone();
    for input in &inputs[1..] {
        for (out, rhs) in output.chunks_exact_mut(2).zip(input.chunks_exact(2)) {
            let lhs = bf16::from_le_bytes(out.try_into().expect("two-byte chunk")).to_f32();
            let rhs = bf16::from_le_bytes(rhs.try_into().expect("two-byte chunk")).to_f32();
            out.copy_from_slice(&bf16::from_f32(float_combine(lhs, rhs, op)).to_le_bytes());
        }
    }
    Ok(output)
}

macro_rules! reduce_integer {
    ($name:ident, $ty:ty, $width:literal) => {
        fn $name(inputs: &[Vec<u8>], op: ReduceOp) -> CommResult<Vec<u8>> {
            let mut output = inputs[0].clone();
            for input in &inputs[1..] {
                for (out, rhs) in output
                    .chunks_exact_mut($width)
                    .zip(input.chunks_exact($width))
                {
                    let lhs = <$ty>::from_le_bytes(out.try_into().expect("integer chunk"));
                    let rhs = <$ty>::from_le_bytes(rhs.try_into().expect("integer chunk"));
                    let value = match op {
                        ReduceOp::Sum => lhs.checked_add(rhs),
                        ReduceOp::Product => lhs.checked_mul(rhs),
                        ReduceOp::Min => Some(lhs.min(rhs)),
                        ReduceOp::Max => Some(lhs.max(rhs)),
                    }
                    .ok_or_else(|| {
                        CommError::Extent(format!(
                            "{} {:?} reduction overflow",
                            stringify!($ty),
                            op
                        ))
                    })?;
                    out.copy_from_slice(&value.to_le_bytes());
                }
            }
            Ok(output)
        }
    };
}

reduce_integer!(reduce_i64, i64, 8);
reduce_integer!(reduce_i32, i32, 4);
reduce_integer!(reduce_u8, u8, 1);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_ops_are_checked_and_deterministic() {
        let inputs = vec![vec![2u8, 7], vec![3, 4], vec![5, 9]];
        assert_eq!(
            reduce_buffers(&inputs, DType::U8, ReduceOp::Product).unwrap(),
            vec![30, 252]
        );
        assert_eq!(
            reduce_buffers(&inputs, DType::U8, ReduceOp::Min).unwrap(),
            vec![2, 4]
        );
        assert_eq!(
            reduce_buffers(&inputs, DType::U8, ReduceOp::Max).unwrap(),
            vec![5, 9]
        );
        assert!(reduce_buffers(&[vec![u8::MAX], vec![1]], DType::U8, ReduceOp::Sum).is_err());
    }

    #[test]
    fn half_reduction_rounds_at_each_rank_in_fixed_order() {
        let f16_inputs: Vec<Vec<u8>> = [0.1f32, 0.2, 0.3]
            .into_iter()
            .map(|value| f16::from_f32(value).to_le_bytes().to_vec())
            .collect();
        let bf16_inputs: Vec<Vec<u8>> = [1.0f32, 2.0, 4.0]
            .into_iter()
            .map(|value| bf16::from_f32(value).to_le_bytes().to_vec())
            .collect();
        assert_eq!(
            reduce_buffers(&f16_inputs, DType::F16, ReduceOp::Sum).unwrap(),
            f16::from_f32(
                f16::from_f32(f16::from_f32(0.1).to_f32() + f16::from_f32(0.2).to_f32()).to_f32()
                    + f16::from_f32(0.3).to_f32()
            )
            .to_le_bytes()
            .to_vec()
        );
        assert_eq!(
            reduce_buffers(&bf16_inputs, DType::BF16, ReduceOp::Product).unwrap(),
            bf16::from_f32(8.0).to_le_bytes().to_vec()
        );
    }
}
