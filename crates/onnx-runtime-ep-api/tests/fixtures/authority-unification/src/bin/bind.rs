use onnx_runtime_ep_api::{ExecutorArtifactConfigTemplate, ExecutorInstanceId};

fn forge(template: ExecutorArtifactConfigTemplate, executor: ExecutorInstanceId) {
    let _forged = template.bind(executor);
}

fn main() {}
