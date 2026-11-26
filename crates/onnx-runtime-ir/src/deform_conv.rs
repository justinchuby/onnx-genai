//! Shared ONNX DeformConv shape and type contract.

use crate::{DataType, Node, Shape, as_static_shape};

pub const INPUTS: [&str; 5] = ["X", "W", "offset", "B", "mask"];
pub const DTYPES: &[DataType] = &[
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
];

pub fn validate_claim(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Result<(), String> {
    let attrs = Attributes::parse(node)?;
    let present = |i: usize| node.inputs.get(i).is_some_and(Option::is_some);
    validate_types(
        opset,
        std::array::from_fn(|i| dtypes.get(i).copied().filter(|_| present(i))),
        None,
    )?;
    for (i, shape) in shapes.iter().take(5).enumerate() {
        // Empty claim shapes also encode unknown rank in the current IR.
        if !present(i) || shape.is_empty() {
            continue;
        }
        let valid = match i {
            3 => shape.len() == 1,
            _ => shape.len() >= 3,
        };
        if !valid {
            return Err(format!("{} has invalid rank {}", INPUTS[i], shape.len()));
        }
    }
    if let Some(x) = shapes.first().filter(|shape| !shape.is_empty()) {
        attrs.validate_rank(x.len() - 2)?;
        for i in [1, 2, 4] {
            if present(i)
                && let Some(s) = shapes.get(i).filter(|shape| !shape.is_empty())
                && s.len() != x.len()
            {
                return Err(format!(
                    "{} rank {} must match X rank {}",
                    INPUTS[i],
                    s.len(),
                    x.len()
                ));
            }
        }
    }
    let concrete: [Option<Vec<usize>>; 5] = std::array::from_fn(|i| {
        shapes
            .get(i)
            .filter(|shape| present(i) && !shape.is_empty())
            .and_then(|s| as_static_shape(s))
    });
    if node
        .inputs
        .iter()
        .enumerate()
        .all(|(i, slot)| slot.is_none() || concrete[i].is_some())
    {
        attrs.resolve(concrete.each_ref().map(Option::as_deref))?;
    }
    Ok(())
}

pub fn validate_types(
    opset: u64,
    types: [Option<DataType>; 5],
    output: Option<DataType>,
) -> Result<(), String> {
    let Some(dtype) = types[0].filter(|&t| t != DataType::Undefined) else {
        return Ok(());
    };
    if opset < 19
        || !match dtype {
            DataType::Float16 | DataType::Float32 | DataType::Float64 => true,
            DataType::BFloat16 => opset >= 22,
            _ => false,
        }
    {
        return Err(format!(
            "X dtype {dtype:?} is unsupported at opset {opset}; use float16, float32 or float64 (bfloat16 requires opset 22+)"
        ));
    }
    for (name, actual) in INPUTS.into_iter().zip(types).chain([("Y", output)]) {
        if let Some(actual) = actual
            && actual != DataType::Undefined
            && actual != dtype
        {
            return Err(format!(
                "{name} dtype {actual:?} must match X dtype {dtype:?}"
            ));
        }
    }
    Ok(())
}

pub struct Attributes {
    pub group: usize,
    pub offset_group: usize,
    kernel: Option<Vec<usize>>,
    strides: Option<Vec<usize>>,
    dilations: Option<Vec<usize>>,
    pads: Option<Vec<usize>>,
}

#[derive(Clone, Copy)]
pub struct Axis {
    pub input: usize,
    pub output: usize,
    pub kernel: usize,
    pub stride: usize,
    pub dilation: usize,
    pub pad: usize,
}

pub struct Geometry {
    pub axes: Vec<Axis>,
    pub output: Vec<usize>,
    pub input_spatial: usize,
    pub output_spatial: usize,
    pub kernel_spatial: usize,
    pub input_channels: usize,
    pub input_per_group: usize,
    pub output_per_group: usize,
    pub input_per_offset_group: usize,
    pub offset_channels: usize,
    pub mask_channels: usize,
}

impl Attributes {
    pub fn parse(node: &Node) -> Result<Self, String> {
        if !(3..=5).contains(&node.inputs.len()) {
            return Err(format!(
                "requires 3 to 5 positional inputs, got {}",
                node.inputs.len()
            ));
        }
        for (i, name) in INPUTS.iter().take(3).enumerate() {
            if node.inputs[i].is_none() {
                return Err(format!("required input {i} ('{name}') is absent"));
            }
        }
        if node.outputs.len() != 1 {
            return Err(format!("requires one output Y, got {}", node.outputs.len()));
        }
        let group = |name| match node.attr(name) {
            None => Ok(1),
            Some(attr) => attr
                .as_int()
                .and_then(|n| usize::try_from(n).ok())
                .filter(|&n| n > 0)
                .ok_or_else(|| format!("{name} must be a positive integer")),
        };
        let list = |name, min| {
            node.attr(name)
                .map(|attr| {
                    attr.as_ints()
                        .ok_or_else(|| format!("{name} must be an integer list"))?
                        .iter()
                        .map(|&n| {
                            usize::try_from(n)
                                .ok()
                                .filter(|&n| n >= min)
                                .ok_or_else(|| format!("{name} values must be >= {min}, got {n}"))
                        })
                        .collect()
                })
                .transpose()
        };
        Ok(Self {
            group: group("group")?,
            offset_group: group("offset_group")?,
            kernel: list("kernel_shape", 1)?,
            strides: list("strides", 1)?,
            dilations: list("dilations", 1)?,
            pads: list("pads", 0)?,
        })
    }

    pub fn validate_rank(&self, rank: usize) -> Result<(), String> {
        if rank == 0 {
            return Err("X, W, and offset require at least one spatial axis".into());
        }
        for (name, values, len) in [
            ("kernel_shape", &self.kernel, rank),
            ("strides", &self.strides, rank),
            ("dilations", &self.dilations, rank),
            ("pads", &self.pads, rank * 2),
        ] {
            if let Some(values) = values
                && values.len() != len
            {
                return Err(format!(
                    "{name} has {} entries; expected {len} for spatial rank {rank}",
                    values.len()
                ));
            }
        }
        Ok(())
    }

    pub fn axis(&self, rank: usize, d: usize, input: usize, weight: usize) -> Result<Axis, String> {
        let kernel = self.kernel.as_ref().map_or(weight, |k| k[d]);
        if kernel == 0 || kernel != weight {
            return Err(format!(
                "kernel_shape axis {d} ({kernel}) must match positive W extent ({weight})"
            ));
        }
        let stride = self.strides.as_ref().map_or(1, |s| s[d]);
        let dilation = self.dilations.as_ref().map_or(1, |s| s[d]);
        let pad = self.pads.as_ref().map_or(0, |p| p[d]);
        let end = self.pads.as_ref().map_or(0, |p| p[d + rank]);
        let overflow = || format!("axis {d} padding/dilation geometry overflows");
        let effective = (kernel - 1)
            .checked_mul(dilation)
            .and_then(|n| n.checked_add(1))
            .ok_or_else(overflow)?;
        let padded = input
            .checked_add(pad)
            .and_then(|n| n.checked_add(end))
            .ok_or_else(overflow)?;
        let output = padded.checked_sub(effective).map_or(0, |n| n / stride + 1);
        Ok(Axis {
            input,
            output,
            kernel,
            stride,
            dilation,
            pad,
        })
    }

    pub fn resolve(&self, shapes: [Option<&[usize]>; 5]) -> Result<Geometry, String> {
        let [Some(x), Some(w), Some(offset), bias, mask] = shapes else {
            return Err("missing required X, W or offset tensor".into());
        };
        let rank = x.len().saturating_sub(2);
        self.validate_rank(rank)?;
        if w.len() != x.len() || offset.len() != x.len() {
            return Err(format!(
                "X, W, and offset must have the same rank; got {x:?}, {w:?}, {offset:?}"
            ));
        }
        if w[1].checked_mul(self.group) != Some(x[1])
            || w[0] % self.group != 0
            || x[1] % self.offset_group != 0
        {
            return Err(format!(
                "X channels {}, W channels [{}, {}] incompatible with group {} and offset_group {}",
                x[1], w[0], w[1], self.group, self.offset_group
            ));
        }
        let axes = (0..rank)
            .map(|d| self.axis(rank, d, x[d + 2], w[d + 2]))
            .collect::<Result<Vec<_>, _>>()?;
        let spatial = axes.iter().map(|a| a.output).collect::<Vec<_>>();
        let kernel = product("kernel", &w[2..])?;
        let mask_channels = product("mask channels", &[self.offset_group, kernel])?;
        let offset_channels = product("offset channels", &[mask_channels, rank])?;
        let expected = |channels| [vec![x[0], channels], spatial.clone()].concat();
        let offset_shape = expected(offset_channels);
        if offset != offset_shape {
            return Err(format!(
                "offset shape {offset:?}, expected {offset_shape:?}"
            ));
        }
        if let Some(mask) = mask {
            let mask_shape = expected(mask_channels);
            if mask != mask_shape {
                return Err(format!("mask shape {mask:?}, expected {mask_shape:?}"));
            }
        }
        if let Some(bias) = bias
            && bias != [w[0]]
        {
            return Err(format!("B shape {bias:?}, expected [{}]", w[0]));
        }
        let output = expected(w[0]);
        let input_spatial = product("input spatial", &x[2..])?;
        let output_spatial = product("output spatial", &spatial)?;
        for (name, shape) in INPUTS
            .into_iter()
            .zip(shapes)
            .chain([("Y", Some(output.as_slice()))])
        {
            if let Some(shape) = shape {
                product(name, shape)?;
            }
        }
        Ok(Geometry {
            axes,
            output,
            input_spatial,
            output_spatial,
            kernel_spatial: kernel,
            input_channels: x[1],
            input_per_group: w[1],
            output_per_group: w[0] / self.group,
            input_per_offset_group: x[1] / self.offset_group,
            offset_channels,
            mask_channels,
        })
    }
}

fn product(name: &str, dims: &[usize]) -> Result<usize, String> {
    dims.iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| format!("{name} shape {dims:?} overflows element count"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{NodeId, ValueId, static_shape};

    fn node() -> Node {
        Node::new(
            NodeId(0),
            "DeformConv",
            (0..3).map(|i| Some(ValueId(i))).collect(),
            vec![ValueId(3)],
        )
    }

    #[test]
    fn claim_rejects_missing_required_slots_and_excess_arity() {
        for missing in 0..3 {
            let mut n = node();
            n.inputs[missing] = None;
            let error = validate_claim(&n, 19, &[], &[]).unwrap_err();
            assert!(
                error.contains(&format!("required input {missing}")),
                "{error}"
            );
        }
        let mut n = node();
        n.inputs.truncate(2);
        assert!(
            validate_claim(&n, 19, &[], &[])
                .unwrap_err()
                .contains("3 to 5")
        );
        n.inputs.resize(6, Some(ValueId(0)));
        assert!(
            validate_claim(&n, 19, &[], &[])
                .unwrap_err()
                .contains("got 6")
        );
    }

    #[test]
    fn unknown_claim_rank_defers_but_concrete_scalar_remains_invalid() {
        let n = node();
        validate_claim(&n, 19, &[vec![], vec![], vec![]], &[DataType::Float32; 3]).unwrap();
        let known_bad = [vec![], static_shape([1, 1]), vec![]];
        assert!(
            validate_claim(&n, 19, &known_bad, &[DataType::Float32; 3])
                .unwrap_err()
                .contains("W has invalid rank 2")
        );
        let attrs = Attributes::parse(&n).unwrap();
        assert!(
            attrs
                .resolve([Some(&[]), Some(&[]), Some(&[]), None, None])
                .is_err()
        );
    }
}
