#![cfg(target_os = "linux")]

//! A100 probe: can independent OS processes share VMM physical memory while
//! containing a fatal protection fault to the request process that caused it?
//!
//! The orchestrator execs a faulting parent helper, which exports a POSIX VMM
//! handle and execs a child helper that imports it. No process forks and then
//! continues using a CUDA context. The poisoned helper is isolated from both
//! this test harness and unrelated CUDA tests.
//!
//! The isolated roles cover four lifecycle edges: work already in flight in a
//! surviving importer, two simultaneous importers when one faults, a replacement
//! importer after the faulting worker exits, and an importer retaining physical
//! memory after the exporting owner releases its handle and terminates. Export
//! and coordination descriptors are checked for CLOEXEC ownership and leaks.

use std::ffi::c_void;
use std::io::{Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::net::UnixStream;
use std::process::{Command, Stdio};

use cudarc::driver::sys as cu;

const MARKER: u8 = 0x5a;
const SAMPLE_BYTES: usize = 4096;

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

.visible .entry spin_store(
    .param .u64 destination,
    .param .u64 cycles
) {
    .reg .pred %waiting;
    .reg .b64 %destination, %cycles, %start, %now, %elapsed;

    ld.param.u64 %destination, [destination];
    ld.param.u64 %cycles, [cycles];
    mov.u64 %start, %clock64;
spin:
    mov.u64 %now, %clock64;
    sub.u64 %elapsed, %now, %start;
    setp.lt.u64 %waiting, %elapsed, %cycles;
    @%waiting bra spin;
    st.global.u8 [%destination], 0x44;
    ret;
}
"#,
    "\0"
);

fn check(call: &str, result: cu::CUresult) {
    assert_eq!(result, cu::CUresult::CUDA_SUCCESS, "{call}: {result:?}");
}

fn allocation_prop(device: cu::CUdevice) -> cu::CUmemAllocationProp {
    let mut prop: cu::CUmemAllocationProp = unsafe { std::mem::zeroed() };
    prop.type_ = cu::CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    // The child exec imports the allocation from this inherited POSIX fd.
    prop.requestedHandleTypes =
        cu::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    prop
}

fn granularity(prop: &cu::CUmemAllocationProp) -> usize {
    let mut bytes = 0;
    check("cuMemGetAllocationGranularity", unsafe {
        cu::cuMemGetAllocationGranularity(
            &mut bytes,
            prop,
            cu::CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED,
        )
    });
    assert_ne!(bytes, 0);
    bytes
}

fn reserve(bytes: usize) -> cu::CUdeviceptr {
    let mut address = 0;
    check("cuMemAddressReserve", unsafe {
        cu::cuMemAddressReserve(&mut address, bytes, 0, 0, 0)
    });
    address
}

fn set_access(address: cu::CUdeviceptr, bytes: usize, device: cu::CUdevice) {
    let mut access: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    access.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = device;
    access.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READ;
    check("cuMemSetAccess(PROT_READ)", unsafe {
        cu::cuMemSetAccess(address, bytes, &access, 1)
    });
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

fn try_device_read(
    address: cu::CUdeviceptr,
    len: usize,
) -> Result<Vec<u8>, (cu::CUresult, cu::CUresult)> {
    let (module, function) = load_function(c"copy_bytes");
    let mut destination = 0;
    check("cuMemAlloc_v2(read destination)", unsafe {
        cu::cuMemAlloc_v2(&mut destination, len)
    });
    let mut source_arg = address;
    let mut destination_arg = destination;
    let mut length_arg = u32::try_from(len).unwrap();
    let mut params = [
        (&mut source_arg as *mut cu::CUdeviceptr).cast(),
        (&mut destination_arg as *mut cu::CUdeviceptr).cast(),
        (&mut length_arg as *mut u32).cast(),
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
    if launch != cu::CUresult::CUDA_SUCCESS {
        return Err((launch, launch));
    }
    let sync = unsafe { cu::cuCtxSynchronize() };
    if sync != cu::CUresult::CUDA_SUCCESS {
        return Err((launch, sync));
    }
    let mut host = vec![0; len];
    check("cuMemcpyDtoH_v2(read destination)", unsafe {
        cu::cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), destination, len)
    });
    check("cuMemFree_v2(read destination)", unsafe {
        cu::cuMemFree_v2(destination)
    });
    check("cuModuleUnload(copy_bytes)", unsafe {
        cu::cuModuleUnload(module)
    });
    Ok(host)
}

fn device_read(address: cu::CUdeviceptr, len: usize) -> Vec<u8> {
    try_device_read(address, len).unwrap_or_else(|(launch, sync)| {
        panic!("device read failed: launch={launch:?}, sync={sync:?}")
    })
}

fn launch_fault(address: cu::CUdeviceptr) -> (cu::CUresult, cu::CUresult, cu::CUmodule) {
    let (module, function) = load_function(c"store_byte");
    let mut destination_arg = address;
    let mut value_arg = 0u32;
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
    (launch, sync, module)
}

#[cfg(target_os = "linux")]
unsafe fn close_fd(fd: RawFd) {
    unsafe extern "C" {
        fn close(fd: RawFd) -> i32;
    }
    let _ = unsafe { close(fd) };
}

fn fresh_work() -> u8 {
    let (module, function) = load_function(c"store_byte");
    let mut dst = 0;
    check("cuMemAlloc fresh", unsafe {
        cu::cuMemAlloc_v2(&mut dst, 1)
    });
    let mut dst_arg = dst;
    let mut value_arg = 0x33u32;
    let mut params = [
        (&mut dst_arg as *mut cu::CUdeviceptr).cast(),
        (&mut value_arg as *mut u32).cast(),
    ];
    check("cuLaunchKernel fresh", unsafe {
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
    check("cuCtxSynchronize fresh", unsafe { cu::cuCtxSynchronize() });
    let mut value = 0u8;
    check("cuMemcpyDtoH fresh", unsafe {
        cu::cuMemcpyDtoH_v2((&mut value as *mut u8).cast(), dst, 1)
    });
    check("cuMemFree fresh", unsafe { cu::cuMemFree_v2(dst) });
    check("cuModuleUnload fresh", unsafe {
        cu::cuModuleUnload(module)
    });
    value
}

fn launch_inflight_work() -> (cu::CUmodule, cu::CUdeviceptr) {
    let (module, function) = load_function(c"spin_store");
    let mut destination = 0;
    check("cuMemAlloc in-flight", unsafe {
        cu::cuMemAlloc_v2(&mut destination, 1)
    });
    let mut destination_arg = destination;
    // About 200 ms on A100: long enough for the other process to fault while
    // this kernel is demonstrably still executing.
    let mut cycles_arg = 300_000_000u64;
    let mut params = [
        (&mut destination_arg as *mut cu::CUdeviceptr).cast(),
        (&mut cycles_arg as *mut u64).cast(),
    ];
    check("cuLaunchKernel in-flight", unsafe {
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
    (module, destination)
}

fn finish_inflight_work(module: cu::CUmodule, destination: cu::CUdeviceptr) -> u8 {
    check("cuCtxSynchronize in-flight", unsafe {
        cu::cuCtxSynchronize()
    });
    let mut value = 0u8;
    check("cuMemcpyDtoH in-flight", unsafe {
        cu::cuMemcpyDtoH_v2((&mut value as *mut u8).cast(), destination, 1)
    });
    check("cuMemFree in-flight", unsafe {
        cu::cuMemFree_v2(destination)
    });
    check("cuModuleUnload in-flight", unsafe {
        cu::cuModuleUnload(module)
    });
    value
}

fn fd_flags(fd: RawFd) -> i32 {
    unsafe extern "C" {
        fn fcntl(fd: RawFd, command: i32, ...) -> i32;
    }
    const F_GETFD: i32 = 1;
    unsafe { fcntl(fd, F_GETFD) }
}

fn clear_cloexec(fd: RawFd) {
    unsafe extern "C" {
        fn fcntl(fd: RawFd, command: i32, ...) -> i32;
    }
    const F_GETFD: i32 = 1;
    const F_SETFD: i32 = 2;
    const FD_CLOEXEC: i32 = 1;
    let flags = unsafe { fcntl(fd, F_GETFD) };
    assert!(flags >= 0);
    assert_eq!(unsafe { fcntl(fd, F_SETFD, flags & !FD_CLOEXEC) }, 0);
    assert_eq!(fd_flags(fd) & FD_CLOEXEC, 0);
}

fn assert_closed(fd: RawFd) {
    assert_eq!(fd_flags(fd), -1, "fd {fd} remained open");
}

fn open_fd_count() -> usize {
    std::fs::read_dir("/proc/self/fd").unwrap().count()
}

fn raw_context(device: cu::CUdevice, label: &str) -> cu::CUcontext {
    let mut context = std::ptr::null_mut();
    check(label, unsafe {
        cu::cuCtxCreate_v4(&mut context, std::ptr::null_mut(), 0, device)
    });
    context
}

fn spawn_role(role: &str, envs: &[(&str, String)]) -> std::process::Child {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.arg("--exact")
        .arg("process_isolation_preserves_shared_vmm_peer")
        .arg("--nocapture")
        .env("MOBIUS_VMM_PROCESS_ROLE", role)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    for (k, v) in envs {
        c.env(k, v);
    }
    c.spawn().unwrap()
}

fn child_role() {
    check("child cuInit", unsafe { cu::cuInit(0) });
    let mut device = 0;
    check("child cuDeviceGet", unsafe {
        cu::cuDeviceGet(&mut device, 0)
    });
    let context = raw_context(device, "child cuCtxCreate_v4");
    let sock_fd: RawFd = std::env::var("MOBIUS_VMM_SOCKET_FD")
        .unwrap()
        .parse()
        .unwrap();
    let mem_fd: RawFd = std::env::var("MOBIUS_VMM_MEMORY_FD")
        .unwrap()
        .parse()
        .unwrap();
    let bytes: usize = std::env::var("MOBIUS_VMM_BYTES").unwrap().parse().unwrap();
    let mut socket = unsafe { UnixStream::from_raw_fd(sock_fd) };
    let report_fd: RawFd = std::env::var("MOBIUS_VMM_REPORT_FD")
        .unwrap()
        .parse()
        .unwrap();
    let mut report = unsafe { UnixStream::from_raw_fd(report_fd) };
    let mut handle = 0;
    check("child import", unsafe {
        cu::cuMemImportFromShareableHandle(
            &mut handle,
            (mem_fd as isize) as *mut c_void,
            cu::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
    });
    unsafe { close_fd(mem_fd) };
    assert_closed(mem_fd);
    let alias = reserve(bytes);
    check("child map", unsafe {
        cu::cuMemMap(alias, bytes, 0, handle, 0)
    });
    set_access(alias, bytes, device);
    let before = device_read(alias, SAMPLE_BYTES);
    assert!(before.iter().all(|&b| b == MARKER));
    eprintln!("child: imported same physical allocation and read marker=0x{MARKER:02x}");
    let (inflight_module, inflight_destination) = launch_inflight_work();
    socket.write_all(b"R").unwrap();
    let mut sig = [0];
    socket.read_exact(&mut sig).unwrap();
    assert_eq!(&sig, b"F");
    assert_eq!(
        socket.read(&mut sig).unwrap(),
        0,
        "faulting owner must exit before child recovery"
    );
    let sync = unsafe { cu::cuCtxSynchronize() };
    eprintln!("child after parent fault: synchronize={sync:?}");
    check("child sync after fault", sync);
    let inflight = finish_inflight_work(inflight_module, inflight_destination);
    let fresh = fresh_work();
    let after = device_read(alias, SAMPLE_BYTES);
    assert_eq!(fresh, 0x33);
    assert_eq!(inflight, 0x44);
    assert_eq!(after, before);
    eprintln!(
        "child after parent fault: in-flight=0x{inflight:02x}, fresh=0x{fresh:02x}, marker=0x{:02x}",
        after[0]
    );
    check("child unmap", unsafe { cu::cuMemUnmap(alias, bytes) });
    check("child VA free", unsafe {
        cu::cuMemAddressFree(alias, bytes)
    });
    check("child release", unsafe { cu::cuMemRelease(handle) });
    check("child destroy", unsafe { cu::cuCtxDestroy_v2(context) });
    report.write_all(b"H").unwrap();
}

fn parent_role() {
    check("parent cuInit", unsafe { cu::cuInit(0) });
    let mut device = 0;
    check("parent cuDeviceGet", unsafe {
        cu::cuDeviceGet(&mut device, 0)
    });
    let context = raw_context(device, "parent cuCtxCreate_v4");
    let prop = allocation_prop(device);
    let bytes = granularity(&prop);
    let mut handle = 0;
    check("parent create", unsafe {
        cu::cuMemCreate(&mut handle, bytes, &prop, 0)
    });
    let alias = reserve(bytes);
    check("parent map", unsafe {
        cu::cuMemMap(alias, bytes, 0, handle, 0)
    });
    let mut rw: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    rw.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    rw.location.id = device;
    rw.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check("parent access RW", unsafe {
        cu::cuMemSetAccess(alias, bytes, &rw, 1)
    });
    let marker = vec![MARKER; bytes];
    check("parent fill", unsafe {
        cu::cuMemcpyHtoD_v2(alias, marker.as_ptr().cast(), bytes)
    });
    check("parent fill sync", unsafe { cu::cuCtxSynchronize() });
    set_access(alias, bytes, device);
    let mut mem_fd: RawFd = -1;
    check("parent export", unsafe {
        cu::cuMemExportToShareableHandle(
            (&mut mem_fd as *mut RawFd).cast(),
            handle,
            cu::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
    });
    const FD_CLOEXEC: i32 = 1;
    assert_ne!(
        fd_flags(mem_fd) & FD_CLOEXEC,
        0,
        "CUDA export fd must default to close-on-exec"
    );
    let (mut socket, child_socket) = UnixStream::pair().unwrap();
    clear_cloexec(mem_fd);
    clear_cloexec(child_socket.as_raw_fd());
    let child_fd = child_socket.as_raw_fd();
    let child = spawn_role(
        "child",
        &[
            ("MOBIUS_VMM_MEMORY_FD", mem_fd.to_string()),
            ("MOBIUS_VMM_SOCKET_FD", child_fd.to_string()),
            ("MOBIUS_VMM_BYTES", bytes.to_string()),
        ],
    );
    drop(child_socket);
    unsafe { close_fd(mem_fd) };
    assert_closed(mem_fd);
    let report_fd: RawFd = std::env::var("MOBIUS_VMM_REPORT_FD")
        .unwrap()
        .parse()
        .unwrap();
    unsafe { close_fd(report_fd) };
    assert_closed(report_fd);
    let mut sig = [0];
    socket.read_exact(&mut sig).unwrap();
    assert_eq!(&sig, b"R");
    let (launch, sync, module) = launch_fault(alias);
    eprintln!("parent st.global fault: launch={launch:?}, sync={sync:?}");
    assert_eq!(launch, cu::CUresult::CUDA_SUCCESS);
    assert_eq!(sync, cu::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS);
    let sticky = unsafe { cu::cuCtxSynchronize() };
    eprintln!("parent poison check={sticky:?}");
    assert_eq!(sticky, cu::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS);
    socket.write_all(b"F").unwrap();
    // Keep the coordination fd open until the OS tears down this process.
    // The child treats EOF as proof that the faulting owner has terminated.
    let _socket_until_process_exit = std::mem::ManuallyDrop::new(socket);
    let unload = unsafe { cu::cuModuleUnload(module) };
    let unmap = unsafe { cu::cuMemUnmap(alias, bytes) };
    let free = unsafe { cu::cuMemAddressFree(alias, bytes) };
    let release = unsafe { cu::cuMemRelease(handle) };
    let destroy = unsafe { cu::cuCtxDestroy_v2(context) };
    eprintln!(
        "parent cleanup: module={unload:?}, unmap={unmap:?}, free={free:?}, release={release:?}, destroy={destroy:?}"
    );
    // The importer deliberately outlives this poisoned owner.
    drop(child);
}

fn imported_worker_setup() -> (
    cu::CUcontext,
    cu::CUmemGenericAllocationHandle,
    cu::CUdeviceptr,
    usize,
    UnixStream,
) {
    check("worker cuInit", unsafe { cu::cuInit(0) });
    let mut device = 0;
    check("worker cuDeviceGet", unsafe {
        cu::cuDeviceGet(&mut device, 0)
    });
    let context = raw_context(device, "worker context");
    let socket_fd: RawFd = std::env::var("MOBIUS_VMM_SOCKET_FD")
        .unwrap()
        .parse()
        .unwrap();
    let memory_fd: RawFd = std::env::var("MOBIUS_VMM_MEMORY_FD")
        .unwrap()
        .parse()
        .unwrap();
    let bytes = std::env::var("MOBIUS_VMM_BYTES").unwrap().parse().unwrap();
    let socket = unsafe { UnixStream::from_raw_fd(socket_fd) };
    let mut handle = 0;
    check("worker import", unsafe {
        cu::cuMemImportFromShareableHandle(
            &mut handle,
            (memory_fd as isize) as *mut c_void,
            cu::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
        )
    });
    unsafe { close_fd(memory_fd) };
    assert_closed(memory_fd);
    let alias = reserve(bytes);
    check("worker map", unsafe {
        cu::cuMemMap(alias, bytes, 0, handle, 0)
    });
    set_access(alias, bytes, device);
    (context, handle, alias, bytes, socket)
}

fn cleanup_imported_worker(
    context: cu::CUcontext,
    handle: cu::CUmemGenericAllocationHandle,
    alias: cu::CUdeviceptr,
    bytes: usize,
) {
    check("worker unmap", unsafe { cu::cuMemUnmap(alias, bytes) });
    check("worker VA free", unsafe {
        cu::cuMemAddressFree(alias, bytes)
    });
    check("worker release", unsafe { cu::cuMemRelease(handle) });
    check("worker destroy", unsafe { cu::cuCtxDestroy_v2(context) });
}

fn fault_worker_role() {
    let (context, handle, alias, bytes, mut socket) = imported_worker_setup();
    let before = device_read(alias, SAMPLE_BYTES);
    assert!(before.iter().all(|&byte| byte == MARKER));
    socket.write_all(b"R").unwrap();
    let mut signal = [0];
    socket.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"G");
    let (launch, sync, module) = launch_fault(alias);
    eprintln!("fault worker: launch={launch:?}, sync={sync:?}");
    assert_eq!(launch, cu::CUresult::CUDA_SUCCESS);
    assert_eq!(sync, cu::CUresult::CUDA_ERROR_ILLEGAL_ADDRESS);
    socket.write_all(b"X").unwrap();
    let unload = unsafe { cu::cuModuleUnload(module) };
    let unmap = unsafe { cu::cuMemUnmap(alias, bytes) };
    let free = unsafe { cu::cuMemAddressFree(alias, bytes) };
    let release = unsafe { cu::cuMemRelease(handle) };
    let destroy = unsafe { cu::cuCtxDestroy_v2(context) };
    eprintln!(
        "fault worker cleanup: module={unload:?}, unmap={unmap:?}, free={free:?}, \
         release={release:?}, destroy={destroy:?}"
    );
}

fn healthy_worker_role() {
    let (context, handle, alias, bytes, mut socket) = imported_worker_setup();
    let before = device_read(alias, SAMPLE_BYTES);
    assert!(before.iter().all(|&byte| byte == MARKER));
    socket.write_all(b"R").unwrap();
    let mut signal = [0];
    socket.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"F");
    check("healthy worker sync", unsafe { cu::cuCtxSynchronize() });
    let fresh = fresh_work();
    let after = device_read(alias, SAMPLE_BYTES);
    assert_eq!(fresh, 0x33);
    assert_eq!(after, before);
    eprintln!(
        "healthy worker after peer fault: fresh=0x{fresh:02x}, marker=0x{:02x}",
        after[0]
    );
    socket.write_all(b"H").unwrap();
    cleanup_imported_worker(context, handle, alias, bytes);
}

fn replacement_worker_role() {
    let (context, handle, alias, bytes, mut socket) = imported_worker_setup();
    let marker = device_read(alias, SAMPLE_BYTES);
    let fresh = fresh_work();
    assert!(marker.iter().all(|&byte| byte == MARKER));
    assert_eq!(fresh, 0x33);
    eprintln!(
        "replacement worker imported after fault worker exit: fresh=0x{fresh:02x}, marker=0x{:02x}",
        marker[0]
    );
    socket.write_all(b"H").unwrap();
    cleanup_imported_worker(context, handle, alias, bytes);
}

fn export_handle_fd(handle: cu::CUmemGenericAllocationHandle) -> RawFd {
    let mut fd = -1;
    check("export handle", unsafe {
        cu::cuMemExportToShareableHandle(
            (&mut fd as *mut RawFd).cast(),
            handle,
            cu::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            0,
        )
    });
    const FD_CLOEXEC: i32 = 1;
    assert_ne!(fd_flags(fd) & FD_CLOEXEC, 0);
    clear_cloexec(fd);
    fd
}

fn spawn_import_worker(
    role: &str,
    memory_fd: RawFd,
    bytes: usize,
) -> (std::process::Child, UnixStream) {
    let (owner_socket, child_socket) = UnixStream::pair().unwrap();
    clear_cloexec(child_socket.as_raw_fd());
    let child_fd = child_socket.as_raw_fd();
    let child = spawn_role(
        role,
        &[
            ("MOBIUS_VMM_MEMORY_FD", memory_fd.to_string()),
            ("MOBIUS_VMM_SOCKET_FD", child_fd.to_string()),
            ("MOBIUS_VMM_BYTES", bytes.to_string()),
        ],
    );
    drop(child_socket);
    (child, owner_socket)
}

fn dual_importer_owner_role() {
    check("dual owner cuInit", unsafe { cu::cuInit(0) });
    let mut device = 0;
    check("dual owner device", unsafe {
        cu::cuDeviceGet(&mut device, 0)
    });
    let context = raw_context(device, "dual owner context");
    let prop = allocation_prop(device);
    let bytes = granularity(&prop);
    let mut handle = 0;
    check("dual owner create", unsafe {
        cu::cuMemCreate(&mut handle, bytes, &prop, 0)
    });
    let alias = reserve(bytes);
    check("dual owner map", unsafe {
        cu::cuMemMap(alias, bytes, 0, handle, 0)
    });
    let mut rw: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    rw.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    rw.location.id = device;
    rw.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check("dual owner RW", unsafe {
        cu::cuMemSetAccess(alias, bytes, &rw, 1)
    });
    let marker = vec![MARKER; bytes];
    check("dual owner fill", unsafe {
        cu::cuMemcpyHtoD_v2(alias, marker.as_ptr().cast(), bytes)
    });
    check("dual owner fill sync", unsafe { cu::cuCtxSynchronize() });
    set_access(alias, bytes, device);

    let baseline = open_fd_count();
    let fd = export_handle_fd(handle);
    let (mut fault_worker, mut fault_socket) = spawn_import_worker("fault-worker", fd, bytes);
    let (mut healthy_worker, mut healthy_socket) = spawn_import_worker("healthy-worker", fd, bytes);
    unsafe { close_fd(fd) };
    assert_closed(fd);
    let mut signal = [0];
    fault_socket.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"R");
    healthy_socket.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"R");
    fault_socket.write_all(b"G").unwrap();
    fault_socket.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"X");
    assert!(fault_worker.wait().unwrap().success());
    healthy_socket.write_all(b"F").unwrap();
    healthy_socket.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"H");
    assert!(healthy_worker.wait().unwrap().success());
    drop(fault_socket);
    drop(healthy_socket);

    let replacement_fd = export_handle_fd(handle);
    let (mut replacement, mut replacement_socket) =
        spawn_import_worker("replacement-worker", replacement_fd, bytes);
    unsafe { close_fd(replacement_fd) };
    assert_closed(replacement_fd);
    replacement_socket.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"H");
    assert!(replacement.wait().unwrap().success());
    drop(replacement_socket);
    assert_eq!(open_fd_count(), baseline);

    check("dual owner unmap", unsafe { cu::cuMemUnmap(alias, bytes) });
    check("dual owner VA free", unsafe {
        cu::cuMemAddressFree(alias, bytes)
    });
    check("dual owner release", unsafe { cu::cuMemRelease(handle) });
    check("dual owner destroy", unsafe {
        cu::cuCtxDestroy_v2(context)
    });
}

fn lifetime_child_role() {
    let (context, handle, alias, bytes, mut owner_socket) = imported_worker_setup();
    let report_fd: RawFd = std::env::var("MOBIUS_VMM_REPORT_FD")
        .unwrap()
        .parse()
        .unwrap();
    let mut report = unsafe { UnixStream::from_raw_fd(report_fd) };
    let before = device_read(alias, SAMPLE_BYTES);
    assert!(before.iter().all(|&byte| byte == MARKER));
    owner_socket.write_all(b"R").unwrap();
    let mut signal = [0];
    assert_eq!(
        owner_socket.read(&mut signal).unwrap(),
        0,
        "owner socket must close when exporting process exits"
    );
    check("lifetime child sync after owner exit", unsafe {
        cu::cuCtxSynchronize()
    });
    let fresh = fresh_work();
    let after = device_read(alias, SAMPLE_BYTES);
    assert_eq!(fresh, 0x33);
    assert_eq!(after, before);
    eprintln!(
        "surviving importer after owner exit: fresh=0x{fresh:02x}, marker=0x{:02x}",
        after[0]
    );
    cleanup_imported_worker(context, handle, alias, bytes);
    report.write_all(b"H").unwrap();
}

fn lifetime_owner_role() {
    check("lifetime owner cuInit", unsafe { cu::cuInit(0) });
    let mut device = 0;
    check("lifetime owner device", unsafe {
        cu::cuDeviceGet(&mut device, 0)
    });
    let context = raw_context(device, "lifetime owner context");
    let prop = allocation_prop(device);
    let bytes = granularity(&prop);
    let mut handle = 0;
    check("lifetime owner create", unsafe {
        cu::cuMemCreate(&mut handle, bytes, &prop, 0)
    });
    let alias = reserve(bytes);
    check("lifetime owner map", unsafe {
        cu::cuMemMap(alias, bytes, 0, handle, 0)
    });
    let mut rw: cu::CUmemAccessDesc = unsafe { std::mem::zeroed() };
    rw.location.type_ = cu::CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE;
    rw.location.id = device;
    rw.flags = cu::CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    check("lifetime owner RW", unsafe {
        cu::cuMemSetAccess(alias, bytes, &rw, 1)
    });
    let marker = vec![MARKER; bytes];
    check("lifetime owner fill", unsafe {
        cu::cuMemcpyHtoD_v2(alias, marker.as_ptr().cast(), bytes)
    });
    check("lifetime owner sync", unsafe { cu::cuCtxSynchronize() });
    set_access(alias, bytes, device);

    let report_fd: RawFd = std::env::var("MOBIUS_VMM_REPORT_FD")
        .unwrap()
        .parse()
        .unwrap();
    let fd = export_handle_fd(handle);
    let (child, mut owner_socket) = spawn_import_worker("lifetime-child", fd, bytes);
    // The report descriptor came from the orchestrator and is forwarded
    // through this exec into the surviving importer.
    unsafe { close_fd(fd) };
    assert_closed(fd);
    let mut signal = [0];
    owner_socket.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"R");

    check("lifetime owner unmap", unsafe {
        cu::cuMemUnmap(alias, bytes)
    });
    check("lifetime owner VA free", unsafe {
        cu::cuMemAddressFree(alias, bytes)
    });
    check("lifetime owner release", unsafe {
        cu::cuMemRelease(handle)
    });
    check("lifetime owner destroy", unsafe {
        cu::cuCtxDestroy_v2(context)
    });
    drop(owner_socket);
    unsafe { close_fd(report_fd) };
    assert_closed(report_fd);
    // Do not wait: the child must remain alive after this owner process exits.
    drop(child);
}

fn restart_role() {
    check("restart cuInit", unsafe { cu::cuInit(0) });
    let mut device = 0;
    check("restart device", unsafe { cu::cuDeviceGet(&mut device, 0) });
    let context = raw_context(device, "restart context");
    let fresh = fresh_work();
    assert_eq!(fresh, 0x33);
    eprintln!("restart process fresh work=0x{fresh:02x}");
    check("restart destroy", unsafe { cu::cuCtxDestroy_v2(context) });
}

#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA A100")]
#[test]
fn process_isolation_preserves_shared_vmm_peer() {
    match std::env::var("MOBIUS_VMM_PROCESS_ROLE").ok().as_deref() {
        Some("child") => return child_role(),
        Some("parent") => return parent_role(),
        Some("fault-worker") => return fault_worker_role(),
        Some("healthy-worker") => return healthy_worker_role(),
        Some("replacement-worker") => return replacement_worker_role(),
        Some("dual-owner") => return dual_importer_owner_role(),
        Some("lifetime-child") => return lifetime_child_role(),
        Some("lifetime-owner") => return lifetime_owner_role(),
        Some("restart") => return restart_role(),
        Some(x) => panic!("bad role {x}"),
        None => {}
    }
    let (mut fault_report, fault_report_child) = UnixStream::pair().unwrap();
    clear_cloexec(fault_report_child.as_raw_fd());
    let fault_report_fd = fault_report_child.as_raw_fd();
    let mut parent = spawn_role(
        "parent",
        &[("MOBIUS_VMM_REPORT_FD", fault_report_fd.to_string())],
    );
    drop(fault_report_child);
    let status = parent.wait().unwrap();
    assert!(status.success(), "parent helper {status}");
    let mut signal = [0];
    fault_report.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"H");
    eprintln!("orchestrator: importer remained healthy after poisoned owner exited");

    let mut dual_owner = spawn_role("dual-owner", &[]);
    let status = dual_owner.wait().unwrap();
    assert!(status.success(), "dual-importer owner helper {status}");
    eprintln!("orchestrator: healthy importer and replacement survived peer-worker fault");

    let (mut report, report_child) = UnixStream::pair().unwrap();
    clear_cloexec(report_child.as_raw_fd());
    let report_fd = report_child.as_raw_fd();
    let mut lifetime_owner = spawn_role(
        "lifetime-owner",
        &[("MOBIUS_VMM_REPORT_FD", report_fd.to_string())],
    );
    drop(report_child);
    let status = lifetime_owner.wait().unwrap();
    assert!(status.success(), "lifetime owner helper {status}");
    let mut signal = [0];
    report.read_exact(&mut signal).unwrap();
    assert_eq!(&signal, b"H");
    eprintln!("orchestrator: importer retained physical allocation after owner exit");

    let mut restart = spawn_role("restart", &[]);
    let status = restart.wait().unwrap();
    assert!(status.success(), "restart helper {status}");
    eprintln!("orchestrator: new process initialized CUDA and ran fresh work");
}
