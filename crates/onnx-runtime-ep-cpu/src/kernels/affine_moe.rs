use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node, Shape};
use onnx_runtime_optimizer::affine_moe::{OP, THRESHOLD};

use super::affine_bank::AffineBank;

pub(crate) struct Factory;

struct BankFormat {
    block: usize,
    zero: u8,
}

struct AffineMoe {
    k: usize,
    alpha: f32,
    beta: f32,
    limit: f32,
    first: BankFormat,
    down: BankFormat,
}

impl KernelFactory for Factory {
    fn create(&self, n: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        AffineMoe::parse(n).map(|config| Box::new(config) as Box<dyn Kernel>)
    }

    fn unsupported_reason(
        &self,
        node: &Node,
        _opset: u64,
        _input_shapes: &[Shape],
        _input_dtypes: &[DataType],
    ) -> Option<String> {
        AffineMoe::parse(node).err().map(|error| error.to_string())
    }
}

impl AffineMoe {
    fn parse(n: &Node) -> Result<Self> {
        let integer = |key| n.attr(key).and_then(Attribute::as_int);
        let positive = |key| {
            integer(key)
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| error(format!("{key} must be a positive INT64")))
        };
        let float = |key| match n.attr(key) {
            Some(Attribute::Float(v)) if v.is_finite() => Ok(*v),
            _ => Err(error(format!("{key} must be a finite FLOAT"))),
        };
        let format = |block, zero| -> Result<BankFormat> {
            Ok(BankFormat {
                block: positive(block)?,
                zero: integer(zero)
                    .and_then(|v| u8::try_from(v).ok())
                    .filter(|&v| v <= 15)
                    .ok_or_else(|| error(format!("{zero} must be a UINT4 value")))?,
            })
        };
        let limit = float("limit")?;
        if limit <= 0.0 {
            return Err(error("limit must be positive"));
        }
        Ok(Self {
            k: positive("k")?,
            alpha: float("alpha")?,
            beta: float("beta")?,
            limit,
            first: format("first_block", "first_zero")?,
            down: format("down_block", "down_zero")?,
        })
    }
}

impl Kernel for AffineMoe {
    fn supports_strided_input(&self, index: usize) -> bool {
        (2..=7).contains(&index)
    }
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let [x, router, w1, s1, z1, w2, s2, z2] = inputs else {
            return Err(error(
                "expected activation, router and two packed expert banks (each with an optional zero-point slot)",
            ));
        };
        let [out] = outputs else {
            return Err(error("expected one FLOAT output"));
        };
        let &[tokens, experts] = router.shape else {
            return Err(error("router must be [tokens, experts]"));
        };
        let &[_, _, packed_intermediate] = w2.shape else {
            return Err(error(
                "second packed bank must be [experts, hidden, packed_intermediate]",
            ));
        };
        let intermediate = packed_intermediate
            .checked_mul(2)
            .ok_or_else(|| error("intermediate width overflows"))?;
        let inter2 = intermediate
            .checked_mul(2)
            .ok_or_else(|| error("interleaved width overflows"))?;
        let Some((&hidden, leading)) = x.shape.split_last() else {
            return Err(error("input has no hidden axis"));
        };
        if !(1..=2).contains(&leading.len())
            || hidden == 0
            || tokens == 0
            || intermediate == 0
            || leading.iter().try_fold(1usize, |a, b| a.checked_mul(*b)) != Some(tokens)
            || self.k > experts
            || x.shape != out.shape
            || [x.dtype, router.dtype, out.dtype]
                .iter()
                .any(|&d| d != DataType::Float32)
        {
            return Err(error(format!(
                "expected FLOAT input and output with matching token geometry; input={:?}, router={:?}, output={:?}, k={}",
                x.shape, router.shape, out.shape, self.k
            )));
        }
        let first = AffineBank::new(
            w1,
            s1,
            (!z1.is_absent()).then_some(z1),
            [experts, inter2, hidden],
            self.first.block,
            self.first.zero,
        )
        .map_err(|cause| error(format!("first expert bank (inputs 2..4): {cause}")))?;
        let down = AffineBank::new(
            w2,
            s2,
            (!z2.is_absent()).then_some(z2),
            [experts, hidden, intermediate],
            self.down.block,
            self.down.zero,
        )
        .map_err(|cause| error(format!("down expert bank (inputs 5..7): {cause}")))?;
        let x = super::to_dense_f32(x)?;
        let scores = super::to_dense_f32(router)?;
        let mut y = vec![0.0; x.len()];
        let mut indices: Vec<_> = (0..experts).collect();
        let mut selected_logits = vec![0.0; self.k];
        let mut probabilities = vec![0.0; self.k];
        let mut activated = vec![0.0; intermediate];
        for ((token, logits), output) in x
            .chunks_exact(hidden)
            .zip(scores.chunks_exact(experts))
            .zip(y.chunks_exact_mut(hidden))
        {
            // TopK on the reversed expert axis picks larger original IDs on ties.
            let order =
                |a: &usize, b: &usize| logits[*b].total_cmp(&logits[*a]).then_with(|| b.cmp(a));
            if self.k < experts {
                indices.select_nth_unstable_by(self.k, order);
            }
            let selected = &mut indices[..self.k];
            selected.sort_unstable_by(order);
            for (p, &id) in selected_logits.iter_mut().zip(selected.iter()) {
                *p = logits[id];
            }
            super::softmax::softmax_rows(&selected_logits, &mut probabilities, 1, self.k);
            for (&expert, &p) in selected.iter().zip(&probabilities) {
                if p.partial_cmp(&THRESHOLD) != Some(std::cmp::Ordering::Greater) {
                    continue;
                }
                for (col, act) in activated.iter_mut().enumerate() {
                    let row = expert * inter2 + col * 2;
                    let gate = first.dot(row, token);
                    // ONNX Min propagates NaN, unlike f32::min.
                    let gate = if gate > self.limit { self.limit } else { gate };
                    let up = first.dot(row + 1, token).clamp(-self.limit, self.limit) + self.beta;
                    let z = self.alpha * gate;
                    let exp = (-z.abs()).exp();
                    let sigmoid = if z >= 0.0 {
                        1.0 / (1.0 + exp)
                    } else {
                        exp / (1.0 + exp)
                    };
                    *act = (gate * sigmoid) * up;
                }
                for (col, sum) in output.iter_mut().enumerate() {
                    *sum += down.dot(expert * hidden + col, &activated) * p;
                }
            }
        }
        super::write_dense_f32(out, &y)
    }
}

fn error(message: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{OP}: {message}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    #[test]
    fn malformed_scale_reports_the_actual_projection_before_output_writes() {
        let kernel = AffineMoe {
            k: 1,
            alpha: 1.,
            beta: 0.,
            limit: 7.,
            first: BankFormat { block: 1, zero: 7 },
            down: BankFormat { block: 1, zero: 7 },
        };
        for first_bad in [true, false] {
            let x = Owned::f32(&[1, 2], &[1., 2.]);
            let router = Owned::f32(&[1, 1], &[0.]);
            let w1 = Owned::u8(&[1, 4, 1], &[0x88; 4]);
            let z1 = Owned::u8(&[1, 4, 1], &[0x77; 4]);
            let w2 = Owned::u8(&[1, 2, 1], &[0x88; 2]);
            let z2 = Owned::u8(&[1, 2, 1], &[0x77; 2]);
            let mut scales1 = [1.; 8];
            let mut scales2 = [1.; 4];
            if first_bad {
                scales1[0] = 0.;
            } else {
                scales2[0] = 0.;
            }
            let s1 = Owned::f32(&[1, 4, 2], &scales1);
            let s2 = Owned::f32(&[1, 2, 2], &scales2);
            let mut out = Owned::f32(&[1, 2], &[123.; 2]);
            let error = kernel
                .execute(
                    &[
                        x.view(),
                        router.view(),
                        w1.view(),
                        s1.view(),
                        z1.view(),
                        w2.view(),
                        s2.view(),
                        z2.view(),
                    ],
                    &mut [out.view_mut()],
                )
                .unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains(if first_bad {
                    "first expert bank (inputs 2..4)"
                } else {
                    "down expert bank (inputs 5..7)"
                }),
                "{message}"
            );
            assert!(message.contains("positive and finite"), "{message}");
            assert_eq!(out.to_f32(), [123.; 2]);
        }
    }
}
