from pathlib import Path
p=Path('/tmp/nxrt-bank-cpu-sample/probe');manifest=Path('/tmp/nxrt-bank-paired/probe/Cargo.toml').read_text().replace('nxrt-bank-paired-probe','nxrt-bank-cpu-sample');(p/'Cargo.toml').write_text(manifest)
s=Path('/tmp/nxrt-bank-paired/probe/src/main.rs').read_text().split('use onnx_runtime_ep_cpu::ExpertBankProbeArm;')[0]
s+='''
use onnx_runtime_ep_cpu::ExpertBankProbeArm;
use std::sync::atomic::{AtomicUsize,Ordering};
use std::io::Write;
fn profile(label: &str, path: &str) {
    let (m,k,n)=(8,128,256);
    let books=Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST,0,64*1024*1024,0)));
    let manager=ProcessMemoryManager::new().unwrap();
    let pool=HostWorkspacePool::governed(&manager,books.clone()).unwrap();
    let resources=CpuMemoryResources::new(DerivedWeightMemory::governed(books.as_ref(),HolderId::new(13)).unwrap(),CpuWeightCacheLimits{dense_bytes:0,expert_bytes:0}).unwrap().with_workspace_pool(pool.clone()).unwrap();
    let kind=match label {"before"|"null"=>ExpertBankProbeArm::Before,"banked"=>ExpertBankProbeArm::Banked,_=>panic!("unknown arm")};
    let creates=Arc::new(AtomicUsize::new(0));
    let ep=Arc::new(CpuExecutionProvider::initialized_with_expert_bank_probe(resources,kind,creates.clone()).unwrap());
    let bytes=model(m,k,n);
    let input=Tensor::from_f32(&[m,k],&(0..m*k).map(|i|(i%31) as f32/1000.-0.015).collect::<Vec<_>>()).unwrap();
    let mut session=InferenceSession::builder().model_bytes(&bytes).option("optimization","none").execution_provider(ep.clone()).build().unwrap();
    let requirement=session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap().bytes;
    let warm=usage().0;while usage().0-warm<0.5 {execute(&mut session,&input);}
    let funded=books.used(Tier::Host);assert_eq!(funded,requirement);assert_eq!(manager.process_used(Tier::Host),requirement);
    let count=creates.load(Ordering::Relaxed);assert!(count>0);
    session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap();
    let out=session.run(&[("a",&input)]).unwrap();assert_eq!(out[0].shape,[m,k]);assert!(out[0].to_vec_f32().iter().all(|x|x.is_finite()));
    let expected=out[0].as_bytes().to_vec();std::fs::write(path,&expected).unwrap();drop(out);
    println!("READY {{\\"arm\\":\\"{label}\\",\\"workspace_bytes\\":{requirement},\\"factory_creates\\":{count}}}");std::io::stdout().flush().unwrap();
    // Sampling window is wall-delimited, not a throughput measurement.
    let start=Instant::now();let mut calls=0;
    while start.elapsed().as_secs_f64()<15. {execute(&mut session,&input);calls+=1;}
    assert_eq!(creates.load(Ordering::Relaxed),count);assert_eq!(books.used(Tier::Host),funded);assert_eq!(manager.process_used(Tier::Host),requirement);
    session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap();let out=session.run(&[("a",&input)]).unwrap();assert_eq!(out[0].as_bytes(),expected);drop(out);
    drop(session);drop(ep);pool.trim_idle(u64::MAX).unwrap();assert_eq!(books.used(Tier::Host),0);
    println!("DONE {{\\"arm\\":\\"{label}\\",\\"calls_in_sampled_window\\":{calls}}}");
}
fn main(){
    let args=std::env::args().skip(1).collect::<Vec<_>>();assert_eq!(args.len(),2);
    rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap().install(||profile(&args[0],&args[1]));
}
''';(p/'src/main.rs').write_text(s)
