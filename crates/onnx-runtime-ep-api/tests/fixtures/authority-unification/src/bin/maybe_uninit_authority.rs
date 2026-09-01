use std::mem::MaybeUninit;

use onnx_runtime_ep_api::ExecutorInstanceId;

fn main() {
    let authority = unsafe { MaybeUninit::uninit().assume_init() };
    let _forged = ExecutorInstanceId::fresh(&authority);
}
