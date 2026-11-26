//! Shared ONNX RNN attribute and tensor contracts for native providers.

use onnx_runtime_ir::{Attribute, DataType, Node};

use crate::{EpError, Result, TensorMut, TensorView};

// RNN-22 added BF16 without changing the RNN-14 computation. Standalone nxrt
// supports it for both revisions; this does not change ONNX schema conformance.
pub const DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
    DataType::Int32,
];

const INPUT_NAMES: [&str; 6] = ["X", "W", "R", "B", "sequence_lens", "initial_h"];

pub fn error(reason: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("RNN: {reason}"))
}

#[derive(Clone, Copy)]
#[repr(i32)]
pub enum ActivationKind {
    Tanh,
    Relu,
    Sigmoid,
    Affine,
    LeakyRelu,
    ThresholdedRelu,
    ScaledTanh,
    HardSigmoid,
    Elu,
    Softsign,
    Softplus,
}

#[derive(Clone, Copy)]
pub struct Activation {
    pub kind: ActivationKind,
    pub alpha: f32,
    pub beta: f32,
}

pub struct Config {
    pub batch_first: bool,
    pub directions: usize,
    pub reverse_first: bool,
    pub activations: [Activation; 2],
    pub clip: Option<f32>,
    hidden: Option<usize>,
}

impl Config {
    pub fn parse(node: &Node) -> Result<Self> {
        let (directions, reverse_first) = match node.attr("direction") {
            None => (1, false),
            Some(attr) => match attr.as_str() {
                Some("forward") => (1, false),
                Some("reverse") => (1, true),
                Some("bidirectional") => (2, false),
                _ => {
                    return Err(error(
                        "direction must be forward, reverse, or bidirectional",
                    ));
                }
            },
        };
        let batch_first = match node.attr("layout") {
            None | Some(Attribute::Int(0)) => false,
            Some(Attribute::Int(1)) => true,
            _ => return Err(error("layout must be 0 (time-first) or 1 (batch-first)")),
        };
        let hidden = match node.attr("hidden_size") {
            None => None,
            Some(Attribute::Int(size)) if *size > 0 => {
                Some(usize::try_from(*size).map_err(|_| error("hidden_size exceeds usize"))?)
            }
            _ => {
                return Err(error(
                    "hidden_size must be a positive integer when specified",
                ));
            }
        };
        let clip = match node.attr("clip") {
            None => None,
            Some(Attribute::Float(value)) if *value >= 0.0 => Some(*value),
            _ => return Err(error("clip must be a nonnegative float")),
        };
        let names = match node.attr("activations") {
            None => None,
            Some(Attribute::Strings(names)) if names.len() == directions => Some(names),
            _ => {
                return Err(error(format!(
                    "activations must contain one name per direction ({directions})"
                )));
            }
        };
        let parameters = |name| match node.attr(name) {
            None => Ok(&[][..]),
            Some(Attribute::Floats(values)) if values.len() <= directions => Ok(values.as_slice()),
            _ => Err(error(format!(
                "{name} must contain at most {directions} floats"
            ))),
        };
        // The lists contain coefficients for activations that consume them,
        // not placeholders for parameterless activations.
        let mut alphas = parameters("activation_alpha")?.iter().copied();
        let mut betas = parameters("activation_beta")?.iter().copied();
        use ActivationKind::*;
        let mut activations = [Activation {
            kind: Tanh,
            alpha: 0.0,
            beta: 0.0,
        }; 2];
        for (dir, activation) in activations.iter_mut().take(directions).enumerate() {
            let name = names.map_or(b"Tanh".as_slice(), |names| names[dir].as_slice());
            let (kind, alpha, beta) = match name {
                b"Tanh" => (Tanh, 0.0, 0.0),
                b"Relu" => (Relu, 0.0, 0.0),
                b"Sigmoid" => (Sigmoid, 0.0, 0.0),
                // Neither Affine nor ScaledTanh has a standalone ONNX
                // schema defining defaults; match ORT's zero coefficients.
                b"Affine" => (Affine, 0.0, 0.0),
                b"LeakyRelu" => (LeakyRelu, 0.01, 0.0),
                b"ThresholdedRelu" => (ThresholdedRelu, 1.0, 0.0),
                b"ScaledTanh" => (ScaledTanh, 0.0, 0.0),
                b"HardSigmoid" => (HardSigmoid, 0.2, 0.5),
                b"Elu" => (Elu, 1.0, 0.0),
                b"Softsign" => (Softsign, 0.0, 0.0),
                b"Softplus" => (Softplus, 0.0, 0.0),
                _ => {
                    return Err(error(format!(
                        "activations[{dir}] is not an ONNX RNN activation: {:?}",
                        String::from_utf8_lossy(name)
                    )));
                }
            };
            *activation = Activation {
                kind,
                alpha: if matches!(
                    kind,
                    Affine | LeakyRelu | ThresholdedRelu | ScaledTanh | HardSigmoid | Elu
                ) {
                    alphas.next().unwrap_or(alpha)
                } else {
                    alpha
                },
                beta: if matches!(kind, Affine | ScaledTanh | HardSigmoid) {
                    betas.next().unwrap_or(beta)
                } else {
                    beta
                },
            };
        }
        Ok(Self {
            batch_first,
            directions,
            reverse_first,
            activations,
            clip,
            hidden,
        })
    }

    pub fn geometry(&self, x: &[usize], r: &[usize]) -> Result<Geometry> {
        let &[first, second, input] = x else {
            return Err(error(format!("X must be rank 3; got {x:?}")));
        };
        let &[_, _, inferred_hidden] = r else {
            return Err(error(format!("R must be rank 3; got {r:?}")));
        };
        let (batch, time) = if self.batch_first {
            (first, second)
        } else {
            (second, first)
        };
        let hidden = self.hidden.unwrap_or(inferred_hidden);
        Ok(Geometry {
            batch,
            time,
            input,
            hidden,
            directions: self.directions,
            batch_first: self.batch_first,
        })
    }

    pub fn validate(&self, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<Geometry> {
        validate_types(inputs.iter().map(|t| {
            if t.is_absent() {
                DataType::Undefined
            } else {
                t.dtype
            }
        }))?;
        if outputs.len() > 2 {
            return Err(error("expected at most two output slots (Y and Y_h)"));
        }
        let g = self.geometry(inputs[0].shape, inputs[2].shape)?;
        let bias = g
            .hidden
            .checked_mul(2)
            .ok_or_else(|| error("2 * hidden_size overflows"))?;
        let state = g.state_shape();
        let expected: [&[usize]; 6] = [
            inputs[0].shape,
            &[g.directions, g.hidden, g.input],
            &[g.directions, g.hidden, g.hidden],
            &[g.directions, bias],
            &[g.batch],
            &state,
        ];
        let check = |name: &str, got: &[usize], expected: &[usize], dtype: DataType| {
            if got != expected {
                return Err(error(format!(
                    "{name} shape must be {expected:?}; got {got:?}"
                )));
            }
            if !got.contains(&0)
                && got
                    .iter()
                    .try_fold(dtype.byte_size(), |p, &d| p.checked_mul(d))
                    .is_none()
            {
                return Err(error(format!(
                    "{name} byte size overflows for shape {got:?}"
                )));
            }
            Ok(())
        };
        for (i, tensor) in inputs.iter().enumerate().filter(|(_, t)| !t.is_absent()) {
            check(INPUT_NAMES[i], tensor.shape, expected[i], tensor.dtype)?;
        }
        for (i, tensor) in outputs.iter().enumerate().filter(|(_, t)| !t.is_absent()) {
            let (name, shape): (_, &[usize]) = if i == 0 {
                ("Y", &g.sequence_shape())
            } else {
                ("Y_h", &state)
            };
            if tensor.dtype != inputs[0].dtype {
                return Err(error(format!(
                    "{name} dtype {:?} must match X dtype {:?}",
                    tensor.dtype, inputs[0].dtype
                )));
            }
            check(name, tensor.shape, shape, tensor.dtype)?;
        }
        Ok(g)
    }
}

pub fn unsupported_reason(node: &Node, dtypes: &[DataType]) -> Option<String> {
    if node.op_type != "RNN" || !onnx_runtime_ir::is_default_domain(&node.domain) {
        return None;
    }
    Config::parse(node)
        .and_then(|_| validate_types(dtypes.iter().copied()))
        .err()
        .map(|err| err.to_string())
}

fn validate_types(dtypes: impl ExactSizeIterator<Item = DataType>) -> Result<()> {
    if !(3..=6).contains(&dtypes.len()) {
        return Err(error(
            "expected 3 required inputs (X, W, R) and up to 3 optional inputs",
        ));
    }
    let mut x = DataType::Undefined;
    for (i, dtype) in dtypes.enumerate() {
        if i == 0 {
            x = dtype;
            if !DTYPES.contains(&x) {
                return Err(error(format!(
                    "X must be Float16, BFloat16, Float32, or Float64; got {x:?}"
                )));
            }
        } else if i < 3 || dtype != DataType::Undefined {
            let expected = if i == 4 { DataType::Int32 } else { x };
            if dtype != expected {
                return Err(error(format!(
                    "{} dtype must be {expected:?}; got {dtype:?}",
                    INPUT_NAMES[i]
                )));
            }
        }
    }
    Ok(())
}

pub fn sequence_length(time: usize, batch: usize, length: i32) -> Result<usize> {
    usize::try_from(length)
        .ok()
        .filter(|&n| n <= time)
        .ok_or_else(|| {
            error(format!(
                "sequence_lens[{batch}] is {length}; expected a value in 0..={time}"
            ))
        })
}

pub struct Geometry {
    pub batch: usize,
    pub time: usize,
    pub input: usize,
    pub hidden: usize,
    pub directions: usize,
    pub batch_first: bool,
}

impl Geometry {
    pub fn state_shape(&self) -> [usize; 3] {
        if self.batch_first {
            [self.batch, self.directions, self.hidden]
        } else {
            [self.directions, self.batch, self.hidden]
        }
    }

    pub fn sequence_shape(&self) -> [usize; 4] {
        if self.batch_first {
            [self.batch, self.time, self.directions, self.hidden]
        } else {
            [self.time, self.directions, self.batch, self.hidden]
        }
    }
}
