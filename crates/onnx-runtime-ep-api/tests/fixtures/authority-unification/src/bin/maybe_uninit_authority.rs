use std::mem::MaybeUninit;

fn main() {
    let _forged =
        MaybeUninit::<onnx_runtime_session::executor::ExecutorArtifactConfig>::uninit();
}
