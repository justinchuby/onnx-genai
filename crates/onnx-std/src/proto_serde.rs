//! Descriptor-driven protobuf textual serialization.

use std::sync::OnceLock;

use onnx_runtime_loader::proto::{FILE_DESCRIPTOR_SET, ModelProto};
use prost::Message;
use prost_reflect::{DescriptorPool, DynamicMessage, MessageDescriptor};

use crate::{Error, Result};

fn model_descriptor() -> MessageDescriptor {
    static POOL: OnceLock<DescriptorPool> = OnceLock::new();
    POOL.get_or_init(|| {
        DescriptorPool::decode(FILE_DESCRIPTOR_SET)
            .expect("the generated ONNX descriptor set must be valid")
    })
    .get_message_by_name("onnx.ModelProto")
    .expect("the bound ONNX schema must define onnx.ModelProto")
}

pub(crate) fn to_dynamic(proto: &ModelProto) -> Result<DynamicMessage> {
    DynamicMessage::decode(model_descriptor(), proto.encode_to_vec().as_slice())
        .map_err(|error| Error::Json(error.to_string()))
}

pub(crate) fn from_dynamic(message: &DynamicMessage) -> Result<ModelProto> {
    ModelProto::decode(message.encode_to_vec().as_slice())
        .map_err(|error| Error::Json(error.to_string()))
}

pub(crate) fn descriptor() -> MessageDescriptor {
    model_descriptor()
}
