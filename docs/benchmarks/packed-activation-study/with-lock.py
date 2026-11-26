#!/usr/bin/env python3
"""Launch this Darwin study under the already established shared host lock."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys

lock_path = Path('/tmp/onnx-genai-hostlock-darwin.flock')
try:
    lock = lock_path.open('r+')
except FileNotFoundError:
    raise SystemExit(f'No shared coordinator lock at {lock_path}; establish that stable file once before starting concurrent tasks, and never unlink or replace it')
with lock:
    if lock.fileno() != 3:
        raise SystemExit('Descriptor 3 is already occupied; launch with-lock.py as a fresh process instead of nesting it inside another lock wrapper')
    print(f'Waiting for shared host lock inode {os.fstat(3).st_ino}', file=sys.stderr, flush=True)
    fcntl.flock(3, fcntl.LOCK_EX)
    if os.fstat(3).st_ino != lock_path.stat().st_ino:
        raise SystemExit('Shared lock was replaced during admission; restore coordination before measuring')
    runner = Path(__file__).resolve().with_name('run.py')
    raise SystemExit(subprocess.run([sys.executable, str(runner), *sys.argv[1:]], pass_fds=(3,)).returncode)
