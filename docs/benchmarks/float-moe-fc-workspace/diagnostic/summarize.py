from pathlib import Path
import json,statistics
root=Path('/tmp/nxrt-float-moe-fc-perf');artifact=root/'diagnostic'
records=json.loads((artifact/'runs.json').read_text())
shapes=[(512,256,256,4,'0','identity'),(512,256,256,4,'0','swiglu2'),(8,64,128,4,'0','silu3')]
assert len(records)==54
code=(root/'driver.py').read_text();code=code[code.index('summary=[]'):].replace('assert len(chosen)==9','assert len(chosen)==18')
exec(code)
