#!/usr/bin/env python3
"""Audit every file:line citation in docs/ARCHITECTURE.md by WHAT IT LANDS ON.

WHY THIS EXISTS, AND WHY check_doc_citations.py DOES NOT COVER IT:
that checker verifies a cited line still EXISTS and repairs it when it moves.
This one asks the different question -- is the cited line EXECUTABLE, or is it
prose? A doc comment inherits the authority of code while carrying none of its
guarantees: nothing runs it, no test covers it, and it drifts silently. A
citation that lands on a docstring looks identical to one that lands on the
assignment 40 lines below, and only one of them is evidence of BEHAVIOUR.

First run: 20 of 95 resolvable citations were anchored on prose, including the
"preemption is disabled" claim -- cited four separate ways, every one of them a
`///` line, while the executable proof sits at batched.rs:759.

CLASSIFIER IS LANGUAGE-AWARE ON PURPOSE. An earlier draft reported 24 because it
treated Rust `#[arg(...)]` attributes and the pointer deref `*slot = start;` as
comments. An instrument built to hunt false authority must not manufacture it.

MUTATION PROVING IT FAILS: point any citation at a `///` line -- e.g. change a
`batched.rs:759` reference to `batched.rs:758` -- and it reports COMMENT.
"""
import re, os, sys
from collections import defaultdict
ROOT="/Users/justinc/Documents/GitHub/onnx-genai-demo"
# argv[1] lets the mutation test run against a COPY. Never mutate the live
# tree to test a checker -- another agent is almost certainly editing it.
DOC=sys.argv[1] if len(sys.argv)>1 else os.path.join(ROOT,"docs/ARCHITECTURE.md")

# index every source file by basename and by suffix path
by_base=defaultdict(list)
for dp,dn,fn in os.walk(ROOT):
    if any(s in dp for s in ("/.git","/node_modules","/target")): continue
    for f in fn:
        if f.endswith((".rs",".js",".py",".sh",".toml")):
            full=os.path.join(dp,f)
            by_base[f].append(full)

doc=open(DOC,encoding="utf-8").read()
cites=re.findall(r'`([A-Za-z0-9_/.-]+\.(?:rs|js|py|sh|toml)):(\d+)(?:-(\d+))?`',doc)

def classify(line, ext):
    s=line.strip()
    if not s: return "BLANK"
    # LANGUAGE-AWARE. Two false-positive classes cost me a wrong headline number:
    #  - Rust/JS `#[arg(...)]` is an ATTRIBUTE: semantic, load-bearing, executable.
    #  - `*slot = start;` is a POINTER DEREF, not a block-comment continuation.
    if ext in (".rs",".js"):
        if s.startswith(("///","//!","//","/*")): return "COMMENT"
        # block-comment continuation is `* ` or bare `*`; `*ident` is a deref
        if s=="*" or s.startswith("* "): return "COMMENT"
    else:
        if s.startswith("#") and not s.startswith("#!"): return "COMMENT"
    if s in ("}","{","};","});",")",");"): return "DELIM"
    return "CODE"

comment_hits=[]; blank=[]; delim=[]; ambiguous=set(); missing=set(); ok=0
seen=set()
for path,a,b in cites:
    key=(path,a,b)
    if key in seen: continue
    seen.add(key)
    base=os.path.basename(path)
    cands=by_base.get(base,[])
    if path.count("/"):
        cands=[c for c in cands if c.endswith(path)]
    if not cands: missing.add(path); continue
    if len(cands)>1: ambiguous.add(path); continue
    f=cands[0]
    lines=open(f,encoding="utf-8",errors="replace").read().split("\n")
    n=int(a)
    if n>len(lines): missing.add(f"{path}:{a} (past EOF)"); continue
    c=classify(lines[n-1], os.path.splitext(f)[1])
    rec=(path,a,b,lines[n-1].strip()[:78])
    if c=="COMMENT": comment_hits.append(rec)
    elif c=="BLANK": blank.append(rec)
    elif c=="DELIM": delim.append(rec)
    else: ok+=1

print(f"unique citations resolved: {ok+len(comment_hits)+len(blank)+len(delim)}  ambiguous(skipped): {len(ambiguous)}  unresolved: {len(missing)}")
print(f"\n### CODE (good): {ok}")
print(f"### COMMENT-POINTING: {len(comment_hits)}")
for p,a,b,t in comment_hits: print(f"  {p}:{a}{'-'+b if b else ''}  ->  {t}")
print(f"### BLANK LINE: {len(blank)}")
for p,a,b,t in blank: print(f"  {p}:{a}")
print(f"### BARE DELIMITER: {len(delim)}")
for p,a,b,t in delim: print(f"  {p}:{a}{'-'+b if b else ''}  ->  {t!r}")
if missing: print("### UNRESOLVED:", sorted(missing)[:10])
