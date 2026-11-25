import pathlib,hashlib,json,subprocess,os,re
root=pathlib.Path('/tmp/nxrt-qmoe-workspace'); p=root/'crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs'; original=p.read_bytes(); source=original.decode(); out=pathlib.Path('/tmp/nxrt-prefetch-controls');out.mkdir(exist_ok=True)
ns={};exec(pathlib.Path('/tmp/nxrt-prefetch-groups-check.py').read_text().split('commands=')[0],ns);env=ns['env']
records=[]
try:
 for name,changed,test,expected in [
 ('wrong-id',source.replace('let leased = load(expert);','let _ = expert; let leased = load(0);'), 'kernels::qmoe::tests::prefetch_group_driver_preserves_sparse_ids_and_two_live_payload_bound',False),
 ('spawn-failure',source.replace('std::thread::Builder::new()\n            .spawn_scoped','std::thread::Builder::new().stack_size(1usize << 62)\n            .spawn_scoped')+'''
#[cfg(test)]
mod prefetch_spawn_failure_probe {
    use super::*;
    #[test]
    fn returns_actionable_error_before_callbacks() {
        let failure = run_prefetched_expert_groups::<()>(
            BTreeMap::from([(3, vec![(0, 0, 1.)])]),
            |_| panic!("loader must not run after spawn failure"),
            |_, _, _| panic!("compute must not run after spawn failure"),
        ).unwrap_err();
        let text = failure.to_string();
        println!("{text}");
        assert!(text.contains("could not start a host worker:"));
        assert!(text.contains(crate::weight_offload::WEIGHT_OFFLOAD_PREFETCH_ENV));
        assert!(text.contains("reduce concurrent CPU work"));
        let EpError::Backend { source, .. } = &failure else { panic!("startup must retain its source"); };
        let startup = source.downcast_ref::<PrefetchWorkerStartError>().unwrap();
        let cause = std::error::Error::source(startup).unwrap();
        assert!(cause.downcast_ref::<std::io::Error>().is_some());
    }
}
''','kernels::qmoe::prefetch_spawn_failure_probe::returns_actionable_error_before_callbacks',True)]:
  assert changed!=source
  p.write_text(changed);(out/(name+'-qmoe.rs')).write_text(changed)
  cmd=['cargo','test','--offline','-p','onnx-runtime-ep-cpu','--lib',test,'--','--exact','--nocapture']
  with (out/(name+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
  log=(out/(name+'.log')).read_text()
  records.append({'name':name,'command':cmd,'exit':r.returncode,'mutated_source_sha256':hashlib.sha256(changed.encode()).hexdigest()})
  (out/'controls.json').write_text(json.dumps(records,indent=2))
  print(name,r.returncode,log[-2200:],flush=True)
  if expected: assert r.returncode==0 and '1 passed; 0 failed' in log and 'could not start a host worker:' in log
  else: assert r.returncode!=0 and '0 passed; 1 failed' in log and 'loaded weights must match the group expert ID' in log
finally:
 p.write_bytes(original)
 assert p.read_bytes()==original
 (out/'restoration.json').write_text(json.dumps({'restored_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},indent=2))
