use onnx_runtime_ep_api::ExecutorInstanceId;

fn main() {
    let authority = Default::default();
    let _forged = ExecutorInstanceId::fresh(&authority);
}
