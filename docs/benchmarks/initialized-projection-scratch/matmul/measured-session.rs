//! Synthetic public InferenceSession strict prepare + run, including output drop.
//! No model throughput claim. Scoped CPU workspace/derived cap is not a process cap.
use std::{hint::black_box, sync::Arc, time::Instant};
use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources, CpuWeightCacheLimits, DerivedWeightMemory};
use onnx_runtime_ir::{Attribute, DataType, DeviceId, Graph, Node, NodeId, TensorData, WeightRef, static_shape};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_memory_api::{DeviceKey, Tier};
use onnx_runtime_memory_governor::{HolderId, HostWorkspacePool, LeaseLedger, LedgerGovernor, MemoryGovernor, ProcessMemoryManager};
use onnx_runtime_session::{InferenceSession, Tensor};
fn usage() -> (f64,u64) {
    let mut raw=std::mem::MaybeUninit::<libc::rusage>::uninit();
    assert_eq!(unsafe{libc::getrusage(libc::RUSAGE_SELF,raw.as_mut_ptr())},0);
    let u=unsafe{raw.assume_init()};
    (u.ru_utime.tv_sec as f64+u.ru_utime.tv_usec as f64/1e6+u.ru_stime.tv_sec as f64+u.ru_stime.tv_usec as f64/1e6,
     if cfg!(target_os="macos"){u.ru_maxrss as u64}else{u.ru_maxrss as u64*1024})
}
fn model(m:usize,k:usize,n:usize)->Vec<u8> {
    let mut graph=Graph::new();
    graph.opset_imports.insert(String::new(),24);
    graph.opset_imports.insert(onnx_runtime_ir::RUNTIME_DOMAIN.into(),1);
    let a=graph.create_named_value("a",DataType::Float32,static_shape([m,k]));
    let y=graph.create_named_value("y",DataType::Float32,static_shape([m,n]));
    let b=graph.create_named_value("b",DataType::Uint8,static_shape([n,k/32,17]));
    let mut packed=vec![0u8;n*(k/32)*17];
    let mut state=0x7a6351d9u32^(n as u32).rotate_left(9)^k as u32;
    for block in packed.chunks_exact_mut(17) {
        state=state.wrapping_mul(1664525).wrapping_add(1013904223);
        block[0]=125+((state>>28)%3) as u8;
        for byte in &mut block[1..] {state=state.wrapping_mul(1664525).wrapping_add(1013904223);*byte=(state>>24) as u8;}
    }
    graph.set_initializer(b,WeightRef::Inline(TensorData::from_raw(DataType::Uint8,vec![n,k/32,17],packed)));
    let mut node=Node::new(NodeId(0),"BlockQuantizedMatMul",vec![Some(a),Some(b)],vec![y]);
    node.domain=onnx_runtime_ir::RUNTIME_DOMAIN.into();
    for (name,value) in [("K",Attribute::Int(k as i64)),("N",Attribute::Int(n as i64)),("format",Attribute::String(b"mxfp4".to_vec()))] {node.attributes.insert(name.into(),value);}
    graph.add_input(a);graph.add_output(y);graph.insert_node(node);
    encode_model(&Model::new(&graph)).unwrap()
}
fn execute(session:&mut InferenceSession,input:&Tensor) {
    session.prepare_with_device_bindings(&[("a",input)],&mut []).unwrap();
    let output=session.run(black_box(&[("a",input)])).unwrap();
    black_box(&output);
    // Output owners are dropped here, before the batch stop time.
}
fn run(m:usize,k:usize,n:usize,reps:usize,path:&str) {
    let limit=64*1024*1024;
    let books=Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST,0,limit,0)));
    let manager=ProcessMemoryManager::new().unwrap();
    let pool=HostWorkspacePool::governed(&manager,books.clone()).unwrap();
    let memory=DerivedWeightMemory::governed(books.as_ref(),HolderId::new(13)).unwrap();
    let resources=CpuMemoryResources::new(memory,CpuWeightCacheLimits{dense_bytes:0,expert_bytes:0}).unwrap().with_workspace_pool(pool.clone()).unwrap();
    let ep=Arc::new(CpuExecutionProvider::initialized_with_resources(resources).unwrap());
    let bytes=model(m,k,n);
    let mut session=InferenceSession::builder().model_bytes(&bytes).option("optimization","none").execution_provider(ep.clone()).build().unwrap();
    let input=Tensor::from_f32(&[m,k],&(0..m*k).map(|i|(i%31) as f32/1000.-0.015).collect::<Vec<_>>()).unwrap();
    let requirement=session.prepare_with_device_bindings(&[("a",&input)],&mut []).unwrap();
    execute(&mut session,&input);
    let warm=usage().0;while usage().0-warm<0.1 {execute(&mut session,&input);}
    let funded=books.used(Tier::Host);
    assert_eq!(funded,manager.process_used(Tier::Host),"only workspace backing, no retained dense weights");
    let mut cpu=Vec::new();let mut wall=Vec::new();
    for _ in 0..9 {
        let start=usage().0;let start_wall=Instant::now();
        for _ in 0..reps {execute(&mut session,&input);}
        cpu.push((usage().0-start)*1000.);wall.push(start_wall.elapsed().as_secs_f64()*1000.);
    }
    let peak=usage().1;
    assert_eq!(books.used(Tier::Host),funded,"warm inference cannot grow retained backing");
    session.prepare_with_device_bindings(&[("a",&input)],&mut []).unwrap();
    let output=session.run(&[("a",&input)]).unwrap();
    std::fs::write(path,output[0].as_bytes()).unwrap();
    assert_eq!(output[0].shape,[m,n]);
    let values=output[0].to_vec_f32();assert!(values.iter().all(|value|value.is_finite()));
    let argmax=values.chunks_exact(n).map(|row|(0..n).max_by(|&a,&b|row[a].total_cmp(&row[b]).then_with(||b.cmp(&a))).unwrap()).collect::<Vec<_>>();
    let mut control_node=Node::new(NodeId(0),"Relu",vec![],vec![]);control_node.domain=String::new();
    let control=ep.get_kernel(&control_node,&[vec![16384]],24).unwrap();
    let control_in=(0..16384).map(|i|(i%31) as f32-15.).collect::<Vec<_>>();let mut control_out=vec![0f32;16384];
    let shape=[16384];let stride=[1];
    let view=TensorView::new(DevicePtr(control_in.as_ptr().cast()),DataType::Float32,&shape,&stride,DeviceId::cpu());
    let mut control_step=|| {
        let result=TensorMut::new(DevicePtrMut(control_out.as_mut_ptr().cast()),DataType::Float32,&shape,&stride,DeviceId::cpu());
        control.execute(black_box(&[view]),black_box(&mut[result])).unwrap();black_box(&control_out);
    };
    for _ in 0..1000 {control_step();}
    let mut controls=Vec::new();
    for _ in 0..9 {let start=usage().0;for _ in 0..100000 {control_step();}controls.push((usage().0-start)*1000.);}
    drop(output);drop(session);drop(ep);pool.trim_idle(u64::MAX).unwrap();assert_eq!(books.used(Tier::Host),0);
    println!("{{\"rows\":{m},\"k\":{k},\"n\":{n},\"reps\":{reps},\"threads\":{},\"cpu_batch_ms\":{cpu:?},\"wall_batch_ms\":{wall:?},\"control_batch_ms\":{controls:?},\"control_reps\":100000,\"workspace_requirement_bytes\":{},\"funded_workspace_bytes\":{funded},\"host_limit_bytes\":{limit},\"absolute_peak_rss_bytes\":{peak},\"argmax\":{argmax:?}}}",rayon::current_num_threads(),requirement.bytes);
}
fn main() {
    let args=std::env::args().skip(1).collect::<Vec<_>>();assert_eq!(args.len(),6,"rows k n reps threads output");
    let numbers=args[..5].iter().map(|s|s.parse::<usize>().unwrap()).collect::<Vec<_>>();
    let(m,k,n,reps,threads)=(numbers[0],numbers[1],numbers[2],numbers[3],numbers[4]);assert!(m>0&&k>0&&k%32==0&&n>0&&reps>0&&threads>0);
    rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap().install(||run(m,k,n,reps,&args[5]));
}
