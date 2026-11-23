//! Observe real, funded execution-workspace backing across concurrent CPU runs.
//!
//! Accumulators are checked out per execution lane, never retained in TLS.
//! `QLINEAR_HOST_BUDGET_MIB` limits actual workspace admission (default 128).
//! `RAYON_NUM_THREADS`, `QLINEAR_M/K/N` and `QLINEAR_ITERS` select the workload.
//! Input/output and other kernel temporaries are outside this workspace ledger;
//! its charged bytes are not a total-RSS prediction. Concurrent refusals are
//! reported explicitly instead of allocating uncharged fallback buffers.
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, TensorMetadata, TensorMut, TensorView,
    WorkspaceView,
};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, DerivedWeightMemory};
use onnx_runtime_ir::{DataType, DeviceId, Node, NodeId};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{
    HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager,
};
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicUsize, Ordering},
};

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).map_or(default, |value| {
        value.parse().unwrap_or_else(|_| {
            panic!("{key}='{value}' is not an unsigned integer; set a nonnegative decimal value")
        })
    })
}
fn view<'a>(
    bytes: &'a [u8],
    dtype: DataType,
    shape: &'a [usize],
    strides: &'a [i64],
) -> TensorView<'a> {
    TensorView::new(
        DevicePtr(bytes.as_ptr().cast()),
        dtype,
        shape,
        strides,
        DeviceId::cpu(),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let threads = env_usize("RAYON_NUM_THREADS", 1);
    let (m, k, n) = (
        env_usize("QLINEAR_M", 768),
        env_usize("QLINEAR_K", 64),
        env_usize("QLINEAR_N", 4096),
    );
    let iters = env_usize("QLINEAR_ITERS", 24);
    if threads == 0 || m == 0 || k == 0 || n == 0 || iters == 0 {
        return Err("the RSS probe needs positive threads, M/K/N and iterations to observe live accumulator backing; set each workload value above zero".into());
    }
    let cap = env_usize("QLINEAR_HOST_BUDGET_MIB", 128)
        .checked_mul(1 << 20)
        .ok_or("QLINEAR_HOST_BUDGET_MIB exceeds addressable bytes; reduce the budget")?;
    let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(
        DeviceKey::HOST,
        0,
        cap as u64,
        0,
    )));
    let manager = ProcessMemoryManager::new()?;
    let workspace = HostWorkspacePool::governed(&manager, books.clone())?;
    let resources = CpuMemoryResources::new(
        DerivedWeightMemory::governed(books.as_ref(), HolderId::new(13))?,
        Default::default(),
    )?
    .with_workspace_pool(workspace.clone())?;
    let ep = Arc::new(CpuExecutionProvider::initialized_with_resources(resources)?);
    let kernel = ep.get_kernel(
        &Node::new(NodeId(0), "QLinearMatMul", vec![], vec![]),
        &[vec![m, k], vec![], vec![], vec![k, n]],
        10,
    )?;
    let (a_shape, b_shape) = ([m, k], [k, n]);
    let (a_strides, b_strides) = ([k as i64, 1], [n as i64, 1]);
    let metadata = [
        TensorMetadata::new(DataType::Uint8, &a_shape, true),
        TensorMetadata::new(DataType::Float32, &[], true),
        TensorMetadata::new(DataType::Uint8, &[], true),
        TensorMetadata::new(DataType::Uint8, &b_shape, true),
        TensorMetadata::new(DataType::Float32, &[], true),
        TensorMetadata::new(DataType::Uint8, &[], true),
        TensorMetadata::new(DataType::Float32, &[], true),
        TensorMetadata::new(DataType::Uint8, &[], true),
    ];
    let request = kernel.workspace_requirement(&metadata)?;
    let a = vec![130u8; m * k];
    let b = vec![120u8; k * n];
    let (a_scale, b_scale, y_scale) = (
        0.5f32.to_le_bytes(),
        0.25f32.to_le_bytes(),
        64f32.to_le_bytes(),
    );
    let (a_zero, b_zero, y_zero) = ([128u8], [127u8], [100u8]);
    let ready = Barrier::new(threads);
    let admitted = AtomicUsize::new(0);
    let refused = AtomicUsize::new(0);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()?;
    pool.broadcast(|_| {
        let kernel = ep.get_kernel(&Node::new(NodeId(0), "QLinearMatMul", vec![], vec![]), &[vec![m,k],vec![],vec![],vec![k,n]], 10)
            .expect("construct this execution lane's QLinearMatMul kernel");
        let allocation=ep.allocate_workspace(request.bytes as usize,request.alignment,request.role);
        // Hold all successful lanes simultaneously to expose the concurrency peak.
        ready.wait();
        let mut allocation=match allocation { Ok(value)=>value,Err(error)=>{refused.fetch_add(1,Ordering::Relaxed);eprintln!("workspace admission: {error}; reduce concurrent threads or raise QLINEAR_HOST_BUDGET_MIB");return;} };
        admitted.fetch_add(1,Ordering::Relaxed);
        let workspace_view=WorkspaceView::new(DevicePtrMut(allocation.as_mut_ptr()),allocation.len());
        let mut output=vec![0u8;m*n];
        for _ in 0..iters {
            let inputs=[view(&a,DataType::Uint8,&a_shape,&a_strides),view(&a_scale,DataType::Float32,&[],&[]),view(&a_zero,DataType::Uint8,&[],&[]),
                view(&b,DataType::Uint8,&b_shape,&b_strides),view(&b_scale,DataType::Float32,&[],&[]),view(&b_zero,DataType::Uint8,&[],&[]),
                view(&y_scale,DataType::Float32,&[],&[]),view(&y_zero,DataType::Uint8,&[],&[])];
            kernel.execute_with_workspace(&inputs,&mut[TensorMut::new(DevicePtrMut(output.as_mut_ptr().cast()),DataType::Uint8,&[m,n],&[n as i64,1],DeviceId::cpu())],Some(workspace_view)).expect("execute the admitted QLinearMatMul lane");
        }
        let accumulated = (-14i32).wrapping_mul(k as i32);
        let expected = ((accumulated as f32 * (0.5 * 0.25 / 64.0)).round_ties_even() as i64)
            .saturating_add(100).clamp(0,255) as u8;
        assert!(output.iter().all(|&value|value==expected), "admitted QLinearMatMul lane produced incorrect output");
        ep.deallocate_workspace(allocation).expect("return completed CPU workspace with its charge");
    });
    let charged = books.used(Tier::Host);
    assert_eq!(charged, manager.process_used(Tier::Host));
    println!(
        "threads={threads} admitted={} refused={} workspace_per_lane={} charged_workspace_bytes={charged} host_limit_bytes={cap}",
        admitted.load(Ordering::Relaxed),
        refused.load(Ordering::Relaxed),
        request.bytes
    );
    std::thread::sleep(std::time::Duration::from_secs(3));
    workspace.trim_idle(u64::MAX)?;
    assert_eq!(books.used(Tier::Host), 0);
    if admitted.load(Ordering::Relaxed) == 0 {
        return Err("no inference lane was admitted; lower concurrency/shape or raise QLINEAR_HOST_BUDGET_MIB".into());
    }
    Ok(())
}
