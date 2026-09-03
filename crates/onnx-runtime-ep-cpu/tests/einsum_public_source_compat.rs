#![deny(warnings)]

use std::{marker::PhantomData, rc::Rc};

use onnx_runtime_ep_cpu::dtype::{ComputeDomain, NumericElem};
use onnx_runtime_ep_cpu::kernels::einsum::{unsupported_reason, unsupported_reason_for_opset};
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, static_shape};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
struct CompatAccumulator<'a> {
    value: u8,
    lifetime: PhantomData<&'a ()>,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl ComputeDomain for CompatAccumulator<'_> {
    fn c_add(self, other: Self) -> Self {
        Self {
            value: self.value.wrapping_add(other.value),
            ..self
        }
    }

    fn c_sub(self, other: Self) -> Self {
        Self {
            value: self.value.wrapping_sub(other.value),
            ..self
        }
    }

    fn c_mul(self, other: Self) -> Self {
        Self {
            value: self.value.wrapping_mul(other.value),
            ..self
        }
    }

    fn c_div(self, other: Self) -> Self {
        Self {
            value: self.value.checked_div(other.value).unwrap_or(0),
            ..self
        }
    }

    fn c_pow(self, other: Self) -> Self {
        Self {
            value: self.value.wrapping_pow(u32::from(other.value)),
            ..self
        }
    }

    fn c_div_usize(self, divisor: usize) -> Self {
        Self {
            value: (usize::from(self.value) / divisor) as u8,
            ..self
        }
    }

    fn c_min(self, other: Self) -> Self {
        Self {
            value: self.value.min(other.value),
            ..self
        }
    }

    fn c_max(self, other: Self) -> Self {
        Self {
            value: self.value.max(other.value),
            ..self
        }
    }
}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct CompatElem<'a> {
    value: u8,
    lifetime: PhantomData<&'a ()>,
    not_send_or_sync: PhantomData<Rc<()>>,
}

impl<'a> NumericElem for CompatElem<'a> {
    const DTYPE: DataType = DataType::Uint8;
    type Acc = CompatAccumulator<'a>;

    fn to_acc(self) -> Self::Acc {
        CompatAccumulator {
            value: self.value,
            lifetime: self.lifetime,
            not_send_or_sync: self.not_send_or_sync,
        }
    }

    fn from_acc(accumulator: Self::Acc) -> Self {
        Self {
            value: accumulator.value,
            lifetime: accumulator.lifetime,
            not_send_or_sync: accumulator.not_send_or_sync,
        }
    }

    fn from_f32_scalar(value: f32) -> Self {
        Self {
            value: value as u8,
            lifetime: PhantomData,
            not_send_or_sync: PhantomData,
        }
    }
}

fn einsum_node() -> Node {
    let mut node = Node::new(NodeId(0), "Einsum", vec![], vec![]);
    node.attributes
        .insert("equation".into(), Attribute::String(b"i->i".to_vec()));
    node
}

#[test]
fn downstream_numeric_elem_keeps_the_pre_einsum_accumulator_contract() {
    fn compile_with_borrowed_non_thread_safe_accumulator<'a>(_borrow: &'a ()) {
        let element = CompatElem {
            value: 6,
            lifetime: PhantomData::<&'a ()>,
            not_send_or_sync: PhantomData,
        };
        let sum = element
            .to_acc()
            .c_add(CompatElem::from_f32_scalar(7.0).to_acc());
        let round_trip = CompatElem::from_acc(sum);

        assert_eq!(round_trip.value, 13);
        assert_eq!(std::mem::size_of_val(&round_trip), 1);
    }

    let borrowed = ();
    compile_with_borrowed_non_thread_safe_accumulator(&borrowed);
}

#[test]
fn legacy_cpu_helper_keeps_its_einsum_12_signature_and_semantics() {
    let reason: Option<String> =
        unsupported_reason(&einsum_node(), &[static_shape([2])], &[DataType::BFloat16]);
    let reason = reason.expect("the legacy helper must apply the Einsum-12 schema");

    assert!(reason.contains("not admitted by Einsum-12"), "{reason}");
}

#[test]
fn schema_aware_cpu_helper_uses_the_explicit_effective_opset() {
    let node = einsum_node();
    let shapes = [static_shape([2])];

    assert!(unsupported_reason_for_opset(&node, 12, &shapes, &[DataType::Float32]).is_none());

    let opset27 = unsupported_reason_for_opset(&node, 27, &shapes, &[DataType::BFloat16]).unwrap();
    assert!(opset27.contains("not admitted by Einsum-12"), "{opset27}");

    assert!(unsupported_reason_for_opset(&node, 28, &shapes, &[DataType::BFloat16]).is_none());
}
