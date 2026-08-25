use cudarc::driver::sys as cu;

const PROBE_PTX: &str = concat!(
    r#"
.version 6.0
.target sm_50
.address_size 64

.visible .entry copy_bytes(
    .param .u64 source,
    .param .u64 destination,
    .param .u32 length
) {
    .reg .pred %done;
    .reg .b32 %index, %length, %value;
    .reg .b64 %source, %destination, %offset, %address;

    ld.param.u64 %source, [source];
    ld.param.u64 %destination, [destination];
    ld.param.u32 %length, [length];
    mov.u32 %index, 0;

loop:
    setp.ge.u32 %done, %index, %length;
    @%done bra copy_complete;
    cvt.u64.u32 %offset, %index;
    add.u64 %address, %source, %offset;
    ld.global.u8 %value, [%address];
    add.u64 %address, %destination, %offset;
    st.global.u8 [%address], %value;
    add.u32 %index, %index, 1;
    bra loop;

copy_complete:
    ret;
}

.visible .entry store_byte(
    .param .u64 destination,
    .param .u32 value
) {
    .reg .b32 %value;
    .reg .b64 %destination;

    ld.param.u64 %destination, [destination];
    ld.param.u32 %value, [value];
    st.global.u8 [%destination], %value;
    ret;
}
"#,
    "\0"
);

fn check(call: &'static str, result: cu::CUresult) {
    assert_eq!(result, cu::CUresult::CUDA_SUCCESS, "{call}: {result:?}");
}

fn load_function(name: &std::ffi::CStr) -> (cu::CUmodule, cu::CUfunction) {
    let mut module = std::ptr::null_mut();
    check("cuModuleLoadData", unsafe {
        cu::cuModuleLoadData(&mut module, PROBE_PTX.as_ptr().cast())
    });
    let mut function = std::ptr::null_mut();
    check("cuModuleGetFunction", unsafe {
        cu::cuModuleGetFunction(&mut function, module, name.as_ptr())
    });
    (module, function)
}

/// Read through the device access path that `cuMemSetAccess` governs.
///
/// A direct DtoH copy is a host copy-engine operation, not the device-location
/// read used by decode kernels. This tiny driver-JITed kernel copies the source
/// bytes into an ordinary RW device allocation before host readback.
pub fn read_through_device(address: cu::CUdeviceptr, len: usize) -> Vec<u8> {
    let (module, function) = load_function(c"copy_bytes");

    let mut destination = 0;
    check("cuMemAlloc_v2", unsafe {
        cu::cuMemAlloc_v2(&mut destination, len)
    });
    let mut source_arg = address;
    let mut destination_arg = destination;
    let mut length_arg = u32::try_from(len).expect("probe length fits u32");
    let mut params = [
        (&mut source_arg as *mut cu::CUdeviceptr).cast(),
        (&mut destination_arg as *mut cu::CUdeviceptr).cast(),
        (&mut length_arg as *mut u32).cast(),
    ];
    check("cuLaunchKernel(copy_bytes)", unsafe {
        cu::cuLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    });
    check("cuCtxSynchronize(copy_bytes)", unsafe {
        cu::cuCtxSynchronize()
    });

    let mut bytes = vec![0u8; len];
    check("cuMemcpyDtoH_v2", unsafe {
        cu::cuMemcpyDtoH_v2(bytes.as_mut_ptr().cast(), destination, len)
    });
    check("cuMemFree_v2", unsafe { cu::cuMemFree_v2(destination) });
    check("cuModuleUnload", unsafe { cu::cuModuleUnload(module) });
    bytes
}

/// Issue a kernel store to a read-only mapping and return both launch and
/// completion results without asserting, so the caller can diagnose whether
/// the protection fault is sticky.
pub fn write_through_device(address: cu::CUdeviceptr, value: u8) -> (cu::CUresult, cu::CUresult) {
    let (module, function) = load_function(c"store_byte");
    let mut destination_arg = address;
    let mut value_arg = u32::from(value);
    let mut params = [
        (&mut destination_arg as *mut cu::CUdeviceptr).cast(),
        (&mut value_arg as *mut u32).cast(),
    ];
    let launch = unsafe {
        cu::cuLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            std::ptr::null_mut(),
            params.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    let sync = unsafe { cu::cuCtxSynchronize() };
    let _ = unsafe { cu::cuModuleUnload(module) };
    (launch, sync)
}
