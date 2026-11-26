use super::{AffineMoe, error, workspace::Workspace};
use crate::kernels::affine_bank;
use onnx_runtime_ep_api::{OutputMetadata, Result, TensorMetadata};
use onnx_runtime_ir::DataType;

pub(super) struct Geometry {
    pub(super) experts: usize,
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
    pub(super) inter2: usize,
    output: usize,
}
impl Geometry {
    pub(super) fn new(
        config: &AffineMoe,
        inputs: &[TensorMetadata<'_>],
        outputs: &[OutputMetadata<'_>],
    ) -> Result<Self> {
        let [x, router, w1, s1, z1, w2, s2, z2] = inputs else {
            return Err(error(format!(
                "expected 8 activation/router/bank slots, received {}; bind both packed banks and optional zero slots",
                inputs.len()
            )));
        };
        if !x.present
            || !router.present
            || x.dtype != DataType::Float32
            || router.dtype != DataType::Float32
            || !(2..=3).contains(&x.shape.len())
        {
            return Err(error(format!(
                "X {:?} {:?} and router {:?} {:?} must be present FLOAT activation rank 2/3 and router rank 2; bind matching floating inputs",
                x.dtype, x.shape, router.dtype, router.shape
            )));
        }
        let &[tokens, experts] = router.shape else {
            return Err(error(format!(
                "router shape {:?} must be [tokens, experts]",
                router.shape
            )));
        };
        let &[_, _, packed_intermediate] = w2.shape else {
            return Err(error(format!(
                "down codes shape {:?} must be [experts,hidden,packed_intermediate]",
                w2.shape
            )));
        };
        let intermediate = packed_intermediate
            .checked_mul(2)
            .ok_or_else(|| error("intermediate width overflows; reduce down bank dimensions"))?;
        let inter2 = intermediate
            .checked_mul(2)
            .ok_or_else(|| error("interleaved width overflows; reduce first bank dimensions"))?;
        let hidden = x.shape[x.shape.len() - 1];
        let leading = x.shape[..x.shape.len() - 1]
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b));
        if tokens == 0
            || hidden == 0
            || intermediate == 0
            || leading != Some(tokens)
            || config.k == 0
            || config.k > experts
        {
            return Err(error(format!(
                "X shape {:?}, router {:?}, intermediate {intermediate}, k={} have invalid token/projection geometry; use positive dimensions, matching tokens and 1<=k<=experts",
                x.shape, router.shape, config.k
            )));
        }
        let output = tokens.checked_mul(hidden).ok_or_else(|| {
            error("output element count overflows; reduce tokens or hidden width")
        })?;
        for (name, count) in [
            ("X/Y", Some(output)),
            ("router", tokens.checked_mul(experts)),
        ] {
            let count = count.ok_or_else(|| {
                error(format!("{name} element count overflows; reduce dimensions"))
            })?;
            std::alloc::Layout::array::<f32>(count).map_err(|_| {
                error(format!(
                    "{name} exceeds addressable FLOAT capacity; reduce dimensions"
                ))
            })?;
        }
        let [out] = outputs else {
            return Err(error(format!(
                "expected one FLOAT Y output, received {} slots",
                outputs.len()
            )));
        };
        match out {
            OutputMetadata::Tensor { dtype, shape }
                if *dtype == DataType::Float32 && shape.as_ref() == x.shape => {}
            OutputMetadata::Unresolved { dtype } if *dtype == DataType::Float32 => {}
            _ => {
                return Err(error(format!(
                    "Y {out:?} must match FLOAT X shape {:?}; bind a matching output",
                    x.shape
                )));
            }
        }
        affine_bank::validate_metadata(
            w1,
            s1,
            z1.present.then_some(z1),
            [experts, inter2, hidden],
            config.first.block,
            config.first.zero,
        )
        .map_err(|e| error(format!("first expert bank (inputs 2..4): {e}")))?;
        affine_bank::validate_metadata(
            w2,
            s2,
            z2.present.then_some(z2),
            [experts, hidden, intermediate],
            config.down.block,
            config.down.zero,
        )
        .map_err(|e| error(format!("down expert bank (inputs 5..7): {e}")))?;
        Ok(Self {
            experts,
            hidden,
            intermediate,
            inter2,
            output,
        })
    }
    pub(super) fn plan(&self, k: usize, x_row: usize, router_row: usize) -> Result<Workspace> {
        Workspace::new(
            self.output,
            x_row,
            router_row,
            k,
            self.intermediate,
            self.experts,
        )
    }
    pub(super) fn conservative(&self, k: usize) -> Result<Workspace> {
        self.plan(k, self.hidden, self.experts)
    }
}
