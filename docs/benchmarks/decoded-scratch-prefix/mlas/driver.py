from pathlib import Path
s=Path('/tmp/nxrt-scratch-prefix-check.py').read_text()
s=s.replace("'/tmp/nxrt-scratch-prefix-validation'", "'/tmp/nxrt-scratch-prefix-mlas'")
s=s.replace("'--offline','-p','onnx-runtime-ep-cpu',", "'--offline','-p','onnx-runtime-ep-cpu','--features','mlas',")
s=s.replace("('oracle',['test','--offline','-p','onnx-runtime-ep-cuda','--test','qmoe_cpu_oracle']),", '')
exec(compile(s,'adapted-prefix-mlas-check','exec'))
