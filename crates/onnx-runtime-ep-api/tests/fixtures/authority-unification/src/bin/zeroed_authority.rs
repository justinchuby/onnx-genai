use onnx_runtime_ep_api::ExecutorInstanceId;

fn main() {
    let authority = unsafe { std::mem::zeroed() };
    let _forged = ExecutorInstanceId::fresh(&authority);
}
