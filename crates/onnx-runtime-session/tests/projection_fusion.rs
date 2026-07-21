use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Command;
use std::sync::{Mutex, MutexGuard};

use onnx_runtime_ir::{DataType, Graph, WeightRef};
use onnx_runtime_session::{InferenceSession, Tensor};

const ENV_NAME: &str = "ONNX_GENAI_PROJECTION_FUSION";
static ENV_LOCK: Mutex<()> = Mutex::new(());

struct ProjectionFusionEnv {
    previous: Option<OsString>,
}

impl ProjectionFusionEnv {
    fn set(enabled: bool) -> Self {
        let previous = std::env::var_os(ENV_NAME);
        // SAFETY: every test in this integration-test process holds ENV_LOCK
        // while mutating or reading this process-global setting.
        unsafe {
            if enabled {
                std::env::set_var(ENV_NAME, "1");
            } else {
                std::env::remove_var(ENV_NAME);
            }
        }
        Self { previous }
    }
}

impl Drop for ProjectionFusionEnv {
    fn drop(&mut self) {
        // SAFETY: ProjectionFusionEnv is created and dropped while ENV_LOCK is
        // held, so no sibling test in this process can race this restoration.
        unsafe {
            if let Some(previous) = &self.previous {
                std::env::set_var(ENV_NAME, previous);
            } else {
                std::env::remove_var(ENV_NAME);
            }
        }
    }
}

fn env_lock() -> MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn model_bytes(scenario: &str) -> Vec<u8> {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/projection_fusion_model.py");
    let output = Command::new("python3")
        .arg(script)
        .arg(scenario)
        .output()
        .expect("run ONNX IR projection model builder");
    assert!(
        output.status.success(),
        "model builder failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    output.stdout
}

fn build(bytes: &[u8], enabled: bool) -> InferenceSession {
    let _env = ProjectionFusionEnv::set(enabled);
    InferenceSession::builder()
        .model_bytes(bytes)
        .build()
        .expect("build projection-fusion session")
}

fn count(graph: &Graph, op_type: &str) -> usize {
    graph
        .nodes
        .values()
        .filter(|node| node.op_type == op_type)
        .count()
}

fn random_input() -> Tensor {
    let mut state = 0x5EED_1234u32;
    let values = (0..32)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect::<Vec<_>>();
    Tensor::from_f32(&[1, 32], &values).unwrap()
}

fn initializer_bytes_by_name(bytes: &[u8], name: &str) -> Vec<u8> {
    let (graph, weights) = onnx_runtime_loader::load_model_bytes_with_weights(bytes, ".").unwrap();
    let (value, weight) = graph
        .initializers
        .iter()
        .find(|(value, _)| graph.value(**value).name.as_deref() == Some(name))
        .expect("named initializer");
    let _ = value;
    weights.bytes(weight).unwrap().to_vec()
}

#[test]
fn projection_fusion_matches_unfused_and_concatenates_rows() {
    let _lock = env_lock();
    let bytes = model_bytes("valid");
    let input = random_input();

    let mut reference = build(&bytes, false);
    let expected = reference.run(&[("X", &input)]).unwrap();

    let mut fused = build(&bytes, true);
    assert_eq!(count(fused.graph(), "MatMulNBits"), 1);
    assert_eq!(count(fused.graph(), "Split"), 1);

    let fused_matmul = fused
        .graph()
        .nodes
        .values()
        .find(|node| node.op_type == "MatMulNBits")
        .unwrap();
    let fused_weight = fused_matmul.inputs[1].unwrap();
    let fused_scales = fused_matmul.inputs[2].unwrap();
    let WeightRef::Inline(fused_weight_data) =
        fused.graph().initializers.get(&fused_weight).unwrap()
    else {
        panic!("fused weight must be materialized inline");
    };
    let WeightRef::Inline(fused_scale_data) =
        fused.graph().initializers.get(&fused_scales).unwrap()
    else {
        panic!("fused scales must be materialized inline");
    };
    assert_eq!(fused_weight_data.dims, vec![4, 1, 16]);
    assert_eq!(fused_scale_data.dims, vec![4, 1]);
    for old_name in ["gate_B", "up_B", "gate_scales", "up_scales"] {
        assert!(
            !fused.graph().initializers.keys().any(|value| fused
                .graph()
                .value(*value)
                .name
                .as_deref()
                == Some(old_name)),
            "orphaned initializer {old_name} must be removed"
        );
    }

    let mut expected_weight = initializer_bytes_by_name(&bytes, "gate_B");
    expected_weight.extend(initializer_bytes_by_name(&bytes, "up_B"));
    assert_eq!(fused_weight_data.data, expected_weight);
    let mut expected_scales = initializer_bytes_by_name(&bytes, "gate_scales");
    expected_scales.extend(initializer_bytes_by_name(&bytes, "up_scales"));
    assert_eq!(fused_scale_data.data, expected_scales);

    let actual = fused.run(&[("X", &input)]).unwrap();
    assert_eq!(actual[0].to_vec_f32(), expected[0].to_vec_f32());
}

#[test]
fn projection_fusion_runs_after_silu_rewrite() {
    let _lock = env_lock();
    let bytes = model_bytes("decomposed");
    let session = build(&bytes, true);
    assert_eq!(count(session.graph(), "Sigmoid"), 0);
    assert_eq!(count(session.graph(), "Silu"), 1);
    assert_eq!(count(session.graph(), "MatMulNBits"), 1);
    assert_eq!(count(session.graph(), "Split"), 1);
}

#[test]
fn projection_fusion_flag_off_is_noop() {
    let _lock = env_lock();
    let bytes = model_bytes("valid");
    let (loaded, _) = onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".").unwrap();
    let session = build(&bytes, false);
    assert_eq!(session.graph().num_nodes(), loaded.num_nodes());
    assert_eq!(count(session.graph(), "MatMulNBits"), 2);
    assert_eq!(count(session.graph(), "Split"), 0);
}

#[test]
fn projection_fusion_skips_bias_zero_point_and_escaping_outputs() {
    let _lock = env_lock();
    for scenario in ["bias", "zero_point", "escape"] {
        let bytes = model_bytes(scenario);
        let session = build(&bytes, true);
        assert_eq!(
            count(session.graph(), "MatMulNBits"),
            2,
            "{scenario} must not fuse"
        );
        assert_eq!(
            count(session.graph(), "Split"),
            0,
            "{scenario} must not split"
        );
    }
}

#[test]
fn projection_fusion_skips_projection_with_external_consumer() {
    let _lock = env_lock();
    let bytes = model_bytes("extra_consumer");
    let (loaded, _) = onnx_runtime_loader::load_model_bytes_with_weights(&bytes, ".").unwrap();
    let session = build(&bytes, true);
    assert_eq!(session.graph().num_nodes(), loaded.num_nodes());
    assert_eq!(count(session.graph(), "MatMulNBits"), 2);
    assert_eq!(count(session.graph(), "Split"), 0);
}

#[test]
fn projection_fusion_materializes_int64_split_initializer() {
    let _lock = env_lock();
    let bytes = model_bytes("valid");
    let session = build(&bytes, true);
    let split = session
        .graph()
        .nodes
        .values()
        .find(|node| node.op_type == "Split")
        .unwrap();
    let split_value = split.inputs[1].expect("explicit Split sizes input");
    assert_eq!(session.graph().value(split_value).dtype, DataType::Int64);
    let WeightRef::Inline(split_data) = session.graph().initializers.get(&split_value).unwrap()
    else {
        panic!("split sizes must be an inline initializer");
    };
    assert_eq!(split_data.dims, vec![2]);
    let sizes = split_data
        .data
        .chunks_exact(8)
        .map(|bytes| i64::from_le_bytes(bytes.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(sizes, vec![2, 2]);
}
