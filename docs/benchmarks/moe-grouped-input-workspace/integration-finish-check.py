import subprocess,pathlib,tempfile,json,hashlib,os
root=pathlib.Path('/tmp/nxrt-compact-wire-grants');os.chdir(root)
os.environ.update(CARGO_TARGET_DIR='/tmp/nxrt-erf-nan-target',CARGO_INCREMENTAL='0',ORT_ROOT='/Users/justinc/Documents/GitHub/onnx-genai/target/debug/build/onnx-genai-ort-sys-14a8134a15d047e2/out/ort-prebuilt')
with pathlib.Path('/tmp/nxrt-moe-gather-integrated-executor.log').open('w') as f:r=subprocess.run(['cargo','test','--offline','-p','onnx-runtime-session','--test','executor','--quiet'],stdout=f,stderr=subprocess.STDOUT)
print(pathlib.Path('/tmp/nxrt-moe-gather-integrated-executor.log').read_text()[-1800:],flush=True)
if r.returncode:raise SystemExit(r.returncode)
proof=json.loads(pathlib.Path('/tmp/nxrt-moe-gather-final-perf/provenance.json').read_text())
patch=root/'docs/benchmarks/moe-grouped-input-workspace/measured-source.patch'
with tempfile.TemporaryDirectory(prefix='nxrt-moe-gather-reproduction-') as temp:
 temp=pathlib.Path(temp)
 for name in proof['source_sha256']:
  p=temp/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(subprocess.check_output(['git','show',f"{proof['base']}:{name}"]))
 subprocess.run(['git','apply',str(patch)],cwd=temp,check=True)
 for name,want in proof['source_sha256'].items():assert hashlib.sha256((temp/name).read_bytes()).hexdigest()==want,name
print('measured source reconstructed from committed patch with exact hashes',flush=True)
