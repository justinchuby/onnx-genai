import fcntl,os,subprocess,time,sys,json
with open('/tmp/onnx-genai-hostlock-darwin.flock','a+') as lock:
 print(json.dumps({'status':'queued','pid':os.getpid(),'action':'Wait for the peer batch on the shared stable flock; no heavy process has started.'}),flush=True)
 fcntl.flock(lock,fcntl.LOCK_EX)
 print(json.dumps({'status':'acquired','pid':os.getpid(),'command':sys.argv[1:]}),flush=True)
 result=subprocess.run(sys.argv[1:],pass_fds=(lock.fileno(),))
 raise SystemExit(result.returncode)
