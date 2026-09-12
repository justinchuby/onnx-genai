//! Schema-only ORT custom operators for the canonical `pkg.nxrt` v1 domain.
//!
//! Execution must always be supplied by the plugin EP. The fallback kernels
//! are deliberately fail-closed so registering the schemas cannot silently
//! move a node to a dense or host implementation.

use std::ffi::{c_char, c_void};
use std::ptr;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ir::block_quant_schema::{
    BLOCK_QUANTIZED_MATMUL_INPUT_COUNT, BLOCK_QUANTIZED_MOE_INPUT_COUNT, BQMM_SCALE,
    BQMOE_FC1_SCALE, BQMOE_FC1_WEIGHT, BQMOE_FC2_SCALE, BQMOE_FC2_WEIGHT, BQMOE_FC3_SCALE,
};

unsafe extern "C" fn create_schema_kernel(
    _op: *const ort::OrtCustomOp,
    _api: *const ort::OrtApi,
    _info: *const ort::OrtKernelInfo,
    kernel: *mut *mut c_void,
) -> *mut ort::OrtStatus {
    if !kernel.is_null() {
        unsafe { *kernel = ptr::dangling_mut::<c_void>() };
    }
    ptr::null_mut()
}

unsafe extern "C" fn schema_fallback_must_not_execute(
    _kernel: *mut c_void,
    _context: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    crate::status::fail_status(
        "pkg.nxrt schema fallback executed; the selected plugin EP must compile this node",
    )
}

unsafe extern "C" fn destroy_schema_kernel(_kernel: *mut c_void) {}

unsafe extern "C" fn provider(_op: *const ort::OrtCustomOp) -> *const c_char {
    c"nxrt_schema_only".as_ptr()
}

unsafe extern "C" fn flexible_type(
    _op: *const ort::OrtCustomOp,
    _index: usize,
) -> ort::ONNXTensorElementDataType {
    ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
}

unsafe extern "C" fn int64_type(
    _op: *const ort::OrtCustomOp,
    _index: usize,
) -> ort::ONNXTensorElementDataType {
    ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64
}

unsafe extern "C" fn bqmm_input_type(
    _op: *const ort::OrtCustomOp,
    index: usize,
) -> ort::ONNXTensorElementDataType {
    if index == BQMM_SCALE {
        ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0
    } else {
        ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
    }
}

unsafe extern "C" fn bqmoe_input_type(
    _op: *const ort::OrtCustomOp,
    index: usize,
) -> ort::ONNXTensorElementDataType {
    if matches!(index, BQMOE_FC1_SCALE | BQMOE_FC2_SCALE | BQMOE_FC3_SCALE) {
        ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E8M0
    } else {
        ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED
    }
}

unsafe extern "C" fn one_output(_op: *const ort::OrtCustomOp) -> usize {
    1
}

unsafe extern "C" fn start_version(_op: *const ort::OrtCustomOp) -> i32 {
    1
}

unsafe extern "C" fn end_version(_op: *const ort::OrtCustomOp) -> i32 {
    i32::MAX
}

unsafe extern "C" fn required(
    _op: *const ort::OrtCustomOp,
    _index: usize,
) -> ort::OrtCustomOpInputOutputCharacteristic {
    ort::INPUT_OUTPUT_REQUIRED
}

unsafe extern "C" fn default_memory(
    _op: *const ort::OrtCustomOp,
    _index: usize,
) -> ort::OrtMemType {
    ort::OrtMemTypeDefault
}

unsafe extern "C" fn zero_arity(_op: *const ort::OrtCustomOp) -> i32 {
    0
}

unsafe extern "C" fn homogeneous(_op: *const ort::OrtCustomOp) -> i32 {
    1
}

unsafe fn release_type_shape(api: *const ort::OrtApi, info: *mut ort::OrtTensorTypeAndShapeInfo) {
    if !info.is_null()
        && let Some(release) = unsafe { (*api).ReleaseTensorTypeAndShapeInfo }
    {
        unsafe { release(info) };
    }
}

unsafe fn infer_output_like_input(
    api: *const ort::OrtApi,
    context: *mut ort::OrtShapeInferContext,
    last_dimension: Option<i64>,
) -> *mut ort::OrtStatus {
    let Some(get_input) = (unsafe { (*api).ShapeInferContext_GetInputTypeShape }) else {
        return crate::status::fail_status("ORT host lacks ShapeInferContext_GetInputTypeShape");
    };
    let Some(get_type) = (unsafe { (*api).GetTensorElementType }) else {
        return crate::status::fail_status("ORT host lacks GetTensorElementType");
    };
    let Some(get_count) = (unsafe { (*api).GetDimensionsCount }) else {
        return crate::status::fail_status("ORT host lacks GetDimensionsCount");
    };
    let Some(get_dimensions) = (unsafe { (*api).GetDimensions }) else {
        return crate::status::fail_status("ORT host lacks GetDimensions");
    };
    let Some(get_symbols) = (unsafe { (*api).GetSymbolicDimensions }) else {
        return crate::status::fail_status("ORT host lacks GetSymbolicDimensions");
    };
    let Some(create_info) = (unsafe { (*api).CreateTensorTypeAndShapeInfo }) else {
        return crate::status::fail_status("ORT host lacks CreateTensorTypeAndShapeInfo");
    };
    let Some(set_type) = (unsafe { (*api).SetTensorElementType }) else {
        return crate::status::fail_status("ORT host lacks SetTensorElementType");
    };
    let Some(set_dimensions) = (unsafe { (*api).SetDimensions }) else {
        return crate::status::fail_status("ORT host lacks SetDimensions");
    };
    let Some(set_symbols) = (unsafe { (*api).SetSymbolicDimensions }) else {
        return crate::status::fail_status("ORT host lacks SetSymbolicDimensions");
    };
    let Some(set_output) = (unsafe { (*api).ShapeInferContext_SetOutputTypeShape }) else {
        return crate::status::fail_status("ORT host lacks ShapeInferContext_SetOutputTypeShape");
    };

    let mut input_info = ptr::null_mut();
    let status = unsafe { get_input(context, 0, &mut input_info) };
    if !status.is_null() {
        return status;
    }
    let mut dtype = ort::ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    let status = unsafe { get_type(input_info, &mut dtype) };
    if !status.is_null() {
        return status;
    }
    let mut rank = 0usize;
    let status = unsafe { get_count(input_info, &mut rank) };
    if !status.is_null() {
        return status;
    }
    if rank == 0 && last_dimension.is_some() {
        return crate::status::fail_status("BlockQuantizedMatMul input A must have rank >= 1");
    }
    let mut dims = vec![0i64; rank];
    let status = unsafe { get_dimensions(input_info, dims.as_mut_ptr(), rank) };
    if !status.is_null() {
        return status;
    }
    let mut input_symbols = vec![ptr::null(); rank];
    let status = unsafe { get_symbols(input_info, input_symbols.as_mut_ptr(), rank) };
    if !status.is_null() {
        return status;
    }
    let symbols = input_symbols
        .iter()
        .map(|symbol| {
            if symbol.is_null() {
                None
            } else {
                Some(unsafe { std::ffi::CStr::from_ptr(*symbol) }.to_owned())
            }
        })
        .collect::<Vec<_>>();
    if let Some(last_dimension) = last_dimension {
        dims[rank - 1] = last_dimension;
    }

    let mut output_info = ptr::null_mut();
    let status = unsafe { create_info(&mut output_info) };
    if !status.is_null() {
        return status;
    }
    let result = (|| {
        let status = unsafe { set_type(output_info, dtype) };
        if !status.is_null() {
            return status;
        }
        let status = unsafe { set_dimensions(output_info, dims.as_ptr(), rank) };
        if !status.is_null() {
            return status;
        }
        let mut output_symbols = symbols
            .iter()
            .map(|symbol| {
                symbol
                    .as_ref()
                    .map_or(ptr::null(), |symbol| symbol.as_ptr())
            })
            .collect::<Vec<_>>();
        if last_dimension.is_some() {
            output_symbols[rank - 1] = ptr::null();
        }
        let status = unsafe { set_symbols(output_info, output_symbols.as_mut_ptr(), rank) };
        if !status.is_null() {
            return status;
        }
        unsafe { set_output(context, 0, output_info) }
    })();
    unsafe { release_type_shape(api, output_info) };
    result
}

unsafe extern "C" fn bqmoe_infer_output(
    _op: *const ort::OrtCustomOp,
    context: *mut ort::OrtShapeInferContext,
) -> *mut ort::OrtStatus {
    let api = crate::status::host_api();
    if api.is_null() {
        return crate::status::fail_status("BlockQuantizedMoE shape inference has no ORT host API");
    }
    unsafe { infer_output_like_input(api, context, None) }
}

unsafe extern "C" fn bqmm_infer_output(
    _op: *const ort::OrtCustomOp,
    context: *mut ort::OrtShapeInferContext,
) -> *mut ort::OrtStatus {
    let api = crate::status::host_api();
    if api.is_null() {
        return crate::status::fail_status(
            "BlockQuantizedMatMul shape inference has no ORT host API",
        );
    }
    let Some(get_attribute) = (unsafe { (*api).ShapeInferContext_GetAttribute }) else {
        return crate::status::fail_status("ORT host lacks ShapeInferContext_GetAttribute");
    };
    let Some(read_attribute) = (unsafe { (*api).ReadOpAttr }) else {
        return crate::status::fail_status("ORT host lacks ReadOpAttr");
    };
    let mut n_attr = ptr::null();
    let status = unsafe { get_attribute(context, c"N".as_ptr(), &mut n_attr) };
    if !status.is_null() {
        return status;
    }
    if n_attr.is_null() {
        return crate::status::fail_status(
            "BlockQuantizedMatMul shape inference requires integer attribute N",
        );
    }
    let mut n = 0i64;
    let mut written = 0usize;
    let status = unsafe {
        read_attribute(
            n_attr,
            ort::ORT_OP_ATTR_INT,
            (&mut n as *mut i64).cast(),
            std::mem::size_of::<i64>(),
            &mut written,
        )
    };
    if !status.is_null() {
        return status;
    }
    if written != std::mem::size_of::<i64>() || n <= 0 {
        return crate::status::fail_status(
            "BlockQuantizedMatMul shape inference requires positive integer attribute N",
        );
    }
    unsafe { infer_output_like_input(api, context, Some(n)) }
}

unsafe extern "C" fn bqmm_name(_op: *const ort::OrtCustomOp) -> *const c_char {
    c"BlockQuantizedMatMul".as_ptr()
}

unsafe extern "C" fn bqmm_input_count(_op: *const ort::OrtCustomOp) -> usize {
    BLOCK_QUANTIZED_MATMUL_INPUT_COUNT
}

unsafe extern "C" fn bqmm_input_characteristic(
    _op: *const ort::OrtCustomOp,
    index: usize,
) -> ort::OrtCustomOpInputOutputCharacteristic {
    if index < BQMM_SCALE {
        ort::INPUT_OUTPUT_REQUIRED
    } else {
        ort::INPUT_OUTPUT_OPTIONAL
    }
}

unsafe extern "C" fn bqmoe_name(_op: *const ort::OrtCustomOp) -> *const c_char {
    c"BlockQuantizedMoE".as_ptr()
}

unsafe extern "C" fn bqmoe_input_count(_op: *const ort::OrtCustomOp) -> usize {
    BLOCK_QUANTIZED_MOE_INPUT_COUNT
}

unsafe extern "C" fn bqmoe_input_characteristic(
    _op: *const ort::OrtCustomOp,
    index: usize,
) -> ort::OrtCustomOpInputOutputCharacteristic {
    if index <= BQMOE_FC1_WEIGHT || index == BQMOE_FC2_WEIGHT {
        ort::INPUT_OUTPUT_REQUIRED
    } else {
        ort::INPUT_OUTPUT_OPTIONAL
    }
}

unsafe extern "C" fn dsa_name(_op: *const ort::OrtCustomOp) -> *const c_char {
    c"DsaIndexSelect".as_ptr()
}

unsafe extern "C" fn dsa_input_count(_op: *const ort::OrtCustomOp) -> usize {
    4
}

macro_rules! schema_op {
    ($name:ident, $get_name:ident, $input_type:ident, $input_count:ident, $input_char:ident, $output_type:ident, $shape_infer:expr) => {
        static $name: ort::OrtCustomOp = ort::OrtCustomOp {
            version: ort::ORT_API_VERSION,
            CreateKernel: None,
            GetName: Some($get_name),
            GetExecutionProviderType: Some(provider),
            GetInputType: Some($input_type),
            GetInputTypeCount: Some($input_count),
            GetOutputType: Some($output_type),
            GetOutputTypeCount: Some(one_output),
            KernelCompute: None,
            KernelDestroy: Some(destroy_schema_kernel),
            GetInputCharacteristic: Some($input_char),
            GetOutputCharacteristic: Some(required),
            GetInputMemoryType: Some(default_memory),
            GetVariadicInputMinArity: Some(zero_arity),
            GetVariadicInputHomogeneity: Some(homogeneous),
            GetVariadicOutputMinArity: Some(zero_arity),
            GetVariadicOutputHomogeneity: Some(homogeneous),
            CreateKernelV2: Some(create_schema_kernel),
            KernelComputeV2: Some(schema_fallback_must_not_execute),
            InferOutputShapeFn: $shape_infer,
            GetStartVersion: Some(start_version),
            GetEndVersion: Some(end_version),
            GetMayInplace: None,
            ReleaseMayInplace: None,
            GetAliasMap: None,
            ReleaseAliasMap: None,
        };
    };
}

schema_op!(
    BLOCK_QUANTIZED_MATMUL,
    bqmm_name,
    bqmm_input_type,
    bqmm_input_count,
    bqmm_input_characteristic,
    flexible_type,
    Some(bqmm_infer_output)
);
schema_op!(
    BLOCK_QUANTIZED_MOE,
    bqmoe_name,
    bqmoe_input_type,
    bqmoe_input_count,
    bqmoe_input_characteristic,
    flexible_type,
    Some(bqmoe_infer_output)
);
schema_op!(
    DSA_INDEX_SELECT,
    dsa_name,
    flexible_type,
    dsa_input_count,
    required,
    int64_type,
    None
);

/// Attach the complete canonical `pkg.nxrt` v1 schema domain to a factory.
///
/// # Safety
/// `factory` and `api` must be live pointers supplied by the ORT plugin host.
pub unsafe fn attach_nxrt_custom_domain(
    factory: *mut ort::OrtEpFactory,
    api: *const ort::OrtApi,
) -> *mut ort::OrtStatus {
    if api.is_null() {
        return crate::status::fail_status("pkg.nxrt custom-op registration has no ORT host API");
    }
    let Some(create_domain) = (unsafe { (*api).CreateCustomOpDomain }) else {
        return crate::status::fail_status("ORT host lacks CreateCustomOpDomain");
    };
    let Some(add) = (unsafe { (*api).CustomOpDomain_Add }) else {
        return crate::status::fail_status("ORT host lacks CustomOpDomain_Add");
    };
    let mut domain = ptr::null_mut();
    let status = unsafe { create_domain(c"pkg.nxrt".as_ptr(), &mut domain) };
    if !status.is_null() {
        return status;
    }
    for descriptor in [
        &BLOCK_QUANTIZED_MATMUL,
        &BLOCK_QUANTIZED_MOE,
        &DSA_INDEX_SELECT,
    ] {
        let status = unsafe { add(domain, descriptor) };
        if !status.is_null() {
            if let Some(release) = unsafe { (*api).ReleaseCustomOpDomain } {
                unsafe { release(domain) };
            }
            return status;
        }
    }
    unsafe { crate::factory::attach_custom_op_domain(factory, domain) };
    ptr::null_mut()
}
