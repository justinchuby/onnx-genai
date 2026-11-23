from pathlib import Path
import shutil
root=Path('/tmp/nxrt-qmoe-workspace');p=Path('/tmp/nxrt-bank-paired/probe');(p/'src').mkdir(parents=True,exist_ok=True)
deps=['onnx-runtime-ep-api','onnx-runtime-ep-cpu','onnx-runtime-ir','onnx-runtime-loader','onnx-runtime-memory-api','onnx-runtime-memory-governor','onnx-runtime-session']
(p/'Cargo.toml').write_text('[package]\nname="nxrt-bank-paired-probe"\nversion="0.0.0"\nedition="2024"\n[dependencies]\nlibc="0.2"\nrayon="1"\n'+''.join(f'{d}={{path="{root}/crates/{d}"'+(',features=["expert_bank_pair_probe"]' if d=='onnx-runtime-ep-cpu' else '')+'}\n' for d in deps))
source=Path('/tmp/nxrt-block-fc-workspace/probe.rs').read_text().split('fn run(m:')[0]
source+='''
use onnx_runtime_ep_cpu::ExpertBankProbeArm;
use std::sync::atomic::{AtomicUsize, Ordering};
struct Arm {
    label: &'static str,
    session: InferenceSession,
    ep: Arc<CpuExecutionProvider>,
    pool: HostWorkspacePool,
    books: Arc<LedgerGovernor>,
    manager: ProcessMemoryManager,
    creates: Arc<AtomicUsize>,
    count: usize,
    requirement: u64,
    funded: u64,
    derived: u64,
    cpu: Vec<f64>, wall: Vec<f64>, controls: Vec<f64>,
}
fn run(m: usize, k: usize, n: usize, reps: usize, path: &str) {
    let dense = std::env::var("NXRT_GATHER_DENSE").ok().as_deref() == Some("1");
    let is_float = std::env::var("NXRT_BANK_OP").ok().as_deref() == Some("float");
    let gate: usize = std::env::var("NXRT_BLOCK_GATED").unwrap().parse().unwrap();
    let rotation: usize = std::env::var("NXRT_PAIR_ROTATION").unwrap_or("0".into()).parse().unwrap();
    let bytes = model(m,k,n);
    let input = Tensor::from_f32(&[m,k], &(0..m*k).map(|i| (i%31) as f32 / 1000. - 0.015).collect::<Vec<_>>()).unwrap();
    let mut arms = Vec::new();
    let mut expected_output = None;
    for j in 0..3 {
        let index = (j+rotation)%3;
        let (label,kind) = [("before",ExpertBankProbeArm::Before),("banked",ExpertBankProbeArm::Banked),("null",ExpertBankProbeArm::Before)][index];
        let limit = 64*1024*1024;
        let books = Arc::new(LedgerGovernor::new(LeaseLedger::new_for_device(DeviceKey::HOST,0,limit,0)));
        let manager = ProcessMemoryManager::new().unwrap();
        let pool = HostWorkspacePool::governed(&manager,books.clone()).unwrap();
        let memory = DerivedWeightMemory::governed(books.as_ref(),HolderId::new(13)).unwrap();
        let resources = CpuMemoryResources::new(memory,CpuWeightCacheLimits{dense_bytes:if dense {limit as usize}else{0},expert_bytes:0}).unwrap().with_workspace_pool(pool.clone()).unwrap();
        let creates = Arc::new(AtomicUsize::new(0));
        let ep = Arc::new(CpuExecutionProvider::initialized_with_expert_bank_probe(resources,kind,creates.clone()).unwrap());
        let mut session = InferenceSession::builder().model_bytes(&bytes).option("optimization","none").execution_provider(ep.clone()).build().unwrap();
        let requirement = session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap().bytes;
        let warm = usage().0;
        while usage().0-warm < 0.1 { execute(&mut session,&input); }
        session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap();
        let out = session.run(&[("a",&input)]).unwrap();
        assert_eq!(out[0].shape,[m,k]);
        assert!(out[0].to_vec_f32().iter().all(|x|x.is_finite()));
        let actual = out[0].as_bytes().to_vec();
        if let Some(expected)= &expected_output { assert_eq!(&actual,expected,"full output mismatch for {label}"); } else { expected_output=Some(actual); }
        drop(out);
        let funded = books.used(Tier::Host);
        assert_eq!(manager.process_used(Tier::Host),requirement);
        let derived = funded-requirement;
        let expected_derived = if dense && !is_float {(if m==1 {2}else{4})*k*n*(if gate!=0 {3}else{2})*4}else{0};
        assert_eq!(derived,expected_derived as u64);
        let count = creates.load(Ordering::Relaxed);
        assert!(count>0,"the selected factory must compile a real kernel");
        arms.push(Arm{label,session,ep,pool,books,manager,creates,count,requirement,funded,derived,cpu:vec![],wall:vec![],controls:vec![]});
    }
    arms.sort_by_key(|a|match a.label {"before"=>0,"banked"=>1,_=>2});
    assert_eq!(arms[0].requirement,arms[2].requirement);
    let widths = if gate!=0 {k.max(n)+k.max(2*n)}else{k+n};
    let delta = if is_float {0}else{m*(widths-k)*4};
    assert_eq!(arms[1].requirement-arms[0].requirement,delta as u64);
    let mut node = Node::new(NodeId(0),"Relu",vec![],vec![]);node.domain=String::new();
    let control=arms[0].ep.get_kernel(&node,&[vec![16384]],24).unwrap();
    let control_in=(0..16384).map(|i|(i%31)as f32-15.).collect::<Vec<_>>();
    let mut control_out=vec![0f32;16384];let shape=[16384];let stride=[1];
    let view=TensorView::new(DevicePtr(control_in.as_ptr().cast()),DataType::Float32,&shape,&stride,DeviceId::cpu());
    let mut control_step=|| {
        let result=TensorMut::new(DevicePtrMut(control_out.as_mut_ptr().cast()),DataType::Float32,&shape,&stride,DeviceId::cpu());
        control.execute(black_box(&[view]),black_box(&mut[result])).unwrap();black_box(&control_out);
    };
    // Excluded pilots calibrate every timed arm to the same baseline iteration count.
    let reps = if reps==0 {
        let start=usage().0;for _ in 0..16 {execute(&mut arms[0].session,&input);}
        (0.250/((usage().0-start)/16.)).ceil().max(1.) as usize
    }else{reps};
    let control_reps = if reps==1 {1}else{
        let start=usage().0;for _ in 0..10000 {control_step();}
        (0.250/((usage().0-start)/10000.)).ceil().max(1.) as usize
    };
    for batch in 0..9 {
        for j in 0..3 {
            let arm=&mut arms[(j+batch+rotation)%3];
            let c=usage().0;for _ in 0..control_reps{control_step();}arm.controls.push((usage().0-c)*1000.);
            let start=usage().0;let wall=Instant::now();
            for _ in 0..reps {execute(&mut arm.session,&input);}
            arm.cpu.push((usage().0-start)*1000.);arm.wall.push(wall.elapsed().as_secs_f64()*1000.);
        }
    }
    std::fs::write(path,expected_output.unwrap()).unwrap();
    for mut arm in arms {
        assert_eq!(arm.creates.load(Ordering::Relaxed),arm.count,"no warm factory dispatch");
        assert_eq!(arm.books.used(Tier::Host),arm.funded);
        assert_eq!(arm.manager.process_used(Tier::Host),arm.requirement);
        arm.session.prepare_with_device_bindings(&[("a",&input)],&mut[]).unwrap();
        let out=arm.session.run(&[("a",&input)]).unwrap();
        assert_eq!(out[0].as_bytes(),std::fs::read(path).unwrap());drop(out);
        println!("{{\\"arm\\":\\"{}\\",\\"reps\\":{reps},\\"control_reps\\":{control_reps},\\"cpu_batch_ms\\":{:?},\\"wall_batch_ms\\":{:?},\\"control_batch_ms\\":{:?},\\"workspace_bytes\\":{},\\"derived_bytes\\":{},\\"factory_creates\\":{}}}",arm.label,arm.cpu,arm.wall,arm.controls,arm.requirement,arm.derived,arm.count);
        drop(arm.session);drop(arm.ep);arm.pool.trim_idle(u64::MAX).unwrap();assert_eq!(arm.books.used(Tier::Host),0);
    }
}
fn main(){
    let args=std::env::args().skip(1).collect::<Vec<_>>();assert_eq!(args.len(),6);
    let v=args[..5].iter().map(|x|x.parse::<usize>().unwrap()).collect::<Vec<_>>();
    assert!(v[0]>0&&v[1]>0&&v[1]%32==0&&v[2]>0&&v[2]%32==0&&v[4]>0);
    rayon::ThreadPoolBuilder::new().num_threads(v[4]).build().unwrap().install(||run(v[0],v[1],v[2],v[3],&args[5]));
}
'''
(p/'src/main.rs').write_text(source)
