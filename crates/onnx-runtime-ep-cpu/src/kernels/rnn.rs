use onnx_runtime_ep_api::rnn::{self, Activation, ActivationKind, Config, Geometry};
use onnx_runtime_ep_api::{Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::dtype::{to_dense, to_dense_f32_widen, write_dense, write_dense_f32_narrow};

pub struct Factory(pub u64);

struct Rnn {
    config: Config,
    since: u64,
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Rnn {
            config: Config::parse(node)?,
            since: self.0,
        }))
    }
}

impl Kernel for Rnn {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let g = self.config.validate(self.since, inputs, outputs)?;
        let lengths = inputs
            .get(4)
            .filter(|t| !t.is_absent())
            .map(to_dense::<i32>)
            .transpose()?
            .map(|lengths| {
                lengths
                    .into_iter()
                    .enumerate()
                    .map(|(batch, length)| rnn::sequence_length(g.time, batch, length))
                    .collect::<Result<Vec<_>>>()
            })
            .transpose()?;
        if g.hidden == 0 || g.batch == 0 || outputs.iter().all(TensorMut::is_absent) {
            return Ok(());
        }
        let requested = |slot| {
            outputs
                .get(slot)
                .is_some_and(|t: &TensorMut| !t.is_absent())
        };
        let (y, yh) = (requested(0), requested(1));
        if inputs[0].dtype == DataType::Float64 {
            let read = |slot| to_dense::<f64>(&inputs[slot]);
            let read_optional = |slot| optional(inputs, slot).map(to_dense::<f64>).transpose();
            let (y, yh) = run_f64(
                &self.config,
                &g,
                &read(0)?,
                &read(1)?,
                &read(2)?,
                read_optional(3)?.as_deref(),
                lengths.as_deref(),
                read_optional(5)?.as_deref(),
                y,
                yh,
            )?;
            if let Some(y) = y {
                write_dense(&mut outputs[0], &y)?;
            }
            if let Some(yh) = yh {
                write_dense(&mut outputs[1], &yh)?;
            }
        } else {
            let read = |slot| to_dense_f32_widen("RNN", &inputs[slot]);
            let read_optional = |slot| {
                optional(inputs, slot)
                    .map(|t| to_dense_f32_widen("RNN", t))
                    .transpose()
            };
            let (y, yh) = run_f32(
                &self.config,
                &g,
                &read(0)?,
                &read(1)?,
                &read(2)?,
                read_optional(3)?.as_deref(),
                lengths.as_deref(),
                read_optional(5)?.as_deref(),
                y,
                yh,
            )?;
            if let Some(y) = y {
                write_dense_f32_narrow("RNN", &mut outputs[0], &y)?;
            }
            if let Some(yh) = yh {
                write_dense_f32_narrow("RNN", &mut outputs[1], &yh)?;
            }
        }
        Ok(())
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn optional<'a, 'b>(inputs: &'a [TensorView<'b>], slot: usize) -> Option<&'a TensorView<'b>> {
    inputs.get(slot).filter(|t| !t.is_absent())
}

fn zeros<T: Copy + Default>(shape: &[usize]) -> Result<Vec<T>> {
    if shape.contains(&0) {
        return Ok(Vec::new());
    }
    let len = shape
        .iter()
        .try_fold(1usize, |p, &d| p.checked_mul(d))
        .ok_or_else(|| rnn::error(format!("CPU scratch size overflows for shape {shape:?}")))?;
    let mut buffer = Vec::new();
    buffer.try_reserve_exact(len).map_err(|e| {
        rnn::error(format!(
            "cannot allocate CPU scratch for shape {shape:?}: {e}"
        ))
    })?;
    buffer.resize(len, T::default());
    Ok(buffer)
}

macro_rules! recurrence {
    ($run:ident, $act:ident, $float:ty) => {
        fn $act(x: $float, activation: Activation) -> $float {
            let (a, b) = (activation.alpha as $float, activation.beta as $float);
            use ActivationKind::*;
            match activation.kind {
                Tanh => x.tanh(),
                Relu => {
                    if x < 0.0 {
                        0.0
                    } else {
                        x
                    }
                }
                Sigmoid => 1.0 / (1.0 + (-x).exp()),
                Affine => a * x + b,
                LeakyRelu => {
                    if x >= 0.0 {
                        x
                    } else {
                        a * x
                    }
                }
                ThresholdedRelu => {
                    if x >= a {
                        x
                    } else {
                        0.0
                    }
                }
                ScaledTanh => a * (b * x).tanh(),
                HardSigmoid => (a * x + b).clamp(0.0, 1.0),
                Elu => {
                    if x >= 0.0 {
                        x
                    } else {
                        a * x.exp_m1()
                    }
                }
                Softsign => {
                    if x.is_infinite() {
                        x.signum()
                    } else {
                        x / (1.0 + x.abs())
                    }
                }
                Softplus => {
                    if x > 0.0 {
                        x + (-x).exp().ln_1p()
                    } else {
                        x.exp().ln_1p()
                    }
                }
            }
        }

        #[allow(clippy::too_many_arguments, clippy::type_complexity)]
        fn $run(
            c: &Config,
            g: &Geometry,
            x: &[$float],
            w: &[$float],
            r: &[$float],
            bias: Option<&[$float]>,
            lengths: Option<&[usize]>,
            initial: Option<&[$float]>,
            want_y: bool,
            want_yh: bool,
        ) -> Result<(Option<Vec<$float>>, Option<Vec<$float>>)> {
            let mut y = want_y.then(|| zeros(&g.sequence_shape())).transpose()?;
            let mut yh = want_yh.then(|| zeros(&g.state_shape())).transpose()?;
            if g.time == 0 {
                return Ok((y, yh));
            }
            let mut prev = zeros::<$float>(&[g.hidden])?;
            let mut next = zeros::<$float>(&[g.hidden])?;
            for dir in 0..g.directions {
                let reverse = c.reverse_first || dir == 1;
                for batch in 0..g.batch {
                    let len = lengths.map_or(g.time, |lengths| lengths[batch]);
                    if len == 0 {
                        continue;
                    }
                    let state = if g.batch_first {
                        batch * g.directions + dir
                    } else {
                        dir * g.batch + batch
                    } * g.hidden;
                    if let Some(initial) = initial {
                        prev.copy_from_slice(&initial[state..state + g.hidden]);
                    } else {
                        prev.fill(0.0);
                    }
                    for step in 0..len {
                        let time = if reverse { len - 1 - step } else { step };
                        let xi = if g.batch_first {
                            batch * g.time + time
                        } else {
                            time * g.batch + batch
                        } * g.input;
                        for (h, target) in next.iter_mut().enumerate() {
                            let row = dir * g.hidden + h;
                            let xw = x[xi..xi + g.input]
                                .iter()
                                .zip(&w[row * g.input..])
                                .fold(0.0, |sum, (&a, &b)| sum + a * b);
                            let hr = prev
                                .iter()
                                .zip(&r[row * g.hidden..])
                                .fold(0.0, |sum, (&a, &b)| sum + a * b);
                            let bias = bias.map_or(0.0, |b| {
                                let i = dir * 2 * g.hidden + h;
                                b[i] + b[i + g.hidden]
                            });
                            let mut value = xw + hr + bias;
                            if let Some(clip) = c.clip {
                                let clip = clip as $float;
                                value = value.clamp(-clip, clip);
                            }
                            *target = $act(value, c.activations[dir]);
                        }
                        if let Some(y) = &mut y {
                            let offset = if g.batch_first {
                                (batch * g.time + time) * g.directions + dir
                            } else {
                                (time * g.directions + dir) * g.batch + batch
                            } * g.hidden;
                            y[offset..offset + g.hidden].copy_from_slice(&next);
                        }
                        std::mem::swap(&mut prev, &mut next);
                    }
                    if let Some(yh) = &mut yh {
                        yh[state..state + g.hidden].copy_from_slice(&prev);
                    }
                }
            }
            Ok((y, yh))
        }
    };
}

recurrence!(run_f32, activation_f32, f32);
recurrence!(run_f64, activation_f64, f64);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuExecutionProvider, kernels::testutil::Owned};
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, NodeId, ValueId, static_shape};

    fn node() -> Node {
        Node::new_with_output_slots(
            NodeId(0),
            "RNN",
            vec![Some(ValueId(0)), Some(ValueId(1)), Some(ValueId(2))],
            vec![None, Some(ValueId(3))],
        )
    }

    #[test]
    fn rnn_pinned_defaults_with_only_final_state() {
        // Pinned ONNX test_simple_rnn_defaults; four identical hidden rows.
        let x = Owned::f32(&[1, 3, 2], &[1., 2., 3., 4., 5., 6.]);
        let w = Owned::f32(&[1, 4, 2], &[0.1; 8]);
        let r = Owned::f32(&[1, 4, 4], &[0.1; 16]);
        let mut unused = Owned::zeros_f32(&[1, 1, 3, 4]);
        let mut yh = Owned::zeros_f32(&[1, 3, 4]);
        let k = CpuExecutionProvider::new()
            .get_kernel(&node(), &[], 19)
            .unwrap();
        k.execute(
            &[x.view(), w.view(), r.view()],
            &mut [unused.view_mut().mark_absent(), yh.view_mut()],
        )
        .unwrap();
        for (&actual, expected) in to_dense::<f32>(&yh.view()).unwrap().iter().zip(
            [0.291_312_63, 0.604_367_8, 0.800_499]
                .into_iter()
                .flat_map(|x| [x; 4]),
        ) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn rnn_consumes_coefficients_only_for_activations_that_use_them() {
        let mut node = node();
        node.attributes.extend([
            (
                "direction".into(),
                Attribute::String(b"bidirectional".to_vec()),
            ),
            (
                "activations".into(),
                Attribute::Strings(vec![b"Relu".to_vec(), b"Affine".to_vec()]),
            ),
            ("activation_alpha".into(), Attribute::Floats(vec![0.5])),
            ("activation_beta".into(), Attribute::Floats(vec![0.25])),
        ]);
        let x = Owned::f32(&[1, 1, 1], &[1.]);
        let w = Owned::f32(&[2, 1, 1], &[1., 1.]);
        let r = Owned::zeros_f32(&[2, 1, 1]);
        let mut unused = Owned::zeros_f32(&[1, 2, 1, 1]);
        let mut yh = Owned::zeros_f32(&[2, 1, 1]);
        let k = CpuExecutionProvider::new()
            .get_kernel(&node, &[], 22)
            .unwrap();
        k.execute(
            &[x.view(), w.view(), r.view()],
            &mut [unused.view_mut().mark_absent(), yh.view_mut()],
        )
        .unwrap();
        assert_eq!(to_dense::<f32>(&yh.view()).unwrap(), [1., 0.75]);
    }

    #[test]
    fn rnn_zero_time_and_opset_dtype_gate() {
        let ep = CpuExecutionProvider::new();
        let mut node = node();
        node.attributes
            .insert("hidden_size".into(), Attribute::Int(1));
        let shapes = [
            static_shape([0, 2, 1]),
            static_shape([1, 1, 1]),
            static_shape([1, 1, 1]),
        ];
        let types = [DataType::BFloat16; 3];
        let rejected = ep.supports_op(&node, 21, &shapes, &types, &[]);
        assert!(rejected.reason().unwrap().contains("opset 22"));
        assert!(
            ep.supports_op(&node, 22, &shapes, &types, &[])
                .is_supported()
        );
        let x = Owned::bf16(&[0, 2, 1], &[]);
        let w = Owned::bf16(&[1, 1, 1], &[1.]);
        let mut yh = Owned::bf16(&[1, 2, 1], &[10., 20.]);
        let mut unused = Owned::zeros(DataType::BFloat16, &[0, 1, 2, 1]);
        let k = ep.get_kernel(&node, &[], 22).unwrap();
        k.execute(
            &[x.view(), w.view(), w.view()],
            &mut [unused.view_mut().mark_absent(), yh.view_mut()],
        )
        .unwrap();
        assert_eq!(yh.bytes, [0; 4]);
    }
}
