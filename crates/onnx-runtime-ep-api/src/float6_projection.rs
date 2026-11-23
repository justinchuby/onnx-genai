use crate::{EpError, Result, TensorMut, TensorView};
use onnx_runtime_ir::DataType;

pub fn geometry(inputs: &[TensorView], outputs: &[TensorMut]) -> Result<(usize, usize)> {
    let bad = || {
        EpError::KernelFailed(
        "Float6Projection requires contiguous FP32 [1,K], FLOAT6 [K,N], and FP32 [1,N] with positive K and N divisible by four; use Cast then MatMul for other layouts".into()
    )
    };
    let ([x, w], [y]) = (inputs, outputs) else {
        return Err(bad());
    };
    let [k, n] = w.shape else { return Err(bad()) };
    if *k == 0
        || *n == 0
        || !n.is_multiple_of(4)
        || k.checked_mul(*n).is_none()
        || k.checked_mul(4).is_none()
        || n.checked_mul(4).is_none()
        || x.shape != [1, *k]
        || y.shape != [1, *n]
        || x.dtype != DataType::Float32
        || y.dtype != DataType::Float32
        || !w.dtype.is_float6()
        || !x.is_contiguous()
        || !w.is_contiguous()
        || !y.is_contiguous()
    {
        return Err(bad());
    }
    x.validate()?;
    w.validate()?;
    y.validate()?;
    Ok((*k, *n))
}
