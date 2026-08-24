"""Read the actual pinned CPU of every decode worker, from /proc, no timing.

Answers one question: does the binary used for the steal A/B place workers one
per physical core (spread, #1729) or two per core on cpus 0-15 (the old compact
mask)? Siblings on this host are (2k, 2k+1), so a spread placement shows only
even CPUs and a compact one shows 0..15 contiguous.
"""
import os, re, subprocess, sys, time, pathlib, collections

BIN = sys.argv[1]
env = dict(os.environ, ONNX_GENAI_CPU_DECODE_WIDTH="16")
p = subprocess.Popen([BIN], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
seen = {}
deadline = time.time() + 25
while time.time() < deadline and len(seen) < 15:
    try:
        for tid in os.listdir(f"/proc/{p.pid}/task"):
            try:
                txt = pathlib.Path(f"/proc/{p.pid}/task/{tid}/status").read_text()
            except OSError:      # ENOENT or ESRCH: the thread is simply gone
                continue
            name = re.search(r"^Name:\s*(.*)$", txt, re.M)
            cpus = re.search(r"^Cpus_allowed_list:\s*(.*)$", txt, re.M)
            if name and cpus and "spmd" in name.group(1):
                seen[tid] = cpus.group(1).strip()
    except OSError:
        break
    time.sleep(0.05)
p.terminate(); p.wait()

vals = sorted(seen.values(), key=lambda v: int(v.split('-')[0]) if v.split('-')[0].isdigit() else 0)
print(f"spmd worker threads found: {len(seen)}")
print("Cpus_allowed_list:", vals)
single = [int(v) for v in vals if v.isdigit()]
if single:
    cores = {c // 2 for c in single}
    print(f"distinct logical cpus = {len(set(single))}, distinct physical cores = {len(cores)}")
    print("VERDICT:", "SPREAD (one worker per physical core)" if len(cores) == len(set(single))
          else "COMPACT (two workers share a core)")
