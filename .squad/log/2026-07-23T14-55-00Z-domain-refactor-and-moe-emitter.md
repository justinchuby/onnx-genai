# 2026-07-23T14:55:00Z — Domain refactor and MoE emitter batch

- Sapper's IR domain-normalization refactor merged to `origin/main` as `06d71ba` + `1073404`; Luv reviewed green twice.
- Leon addressed eleven non-MoE Mobius PR #404 comments at `958f2eb`; Gaff reviewed DSA correctness green.
- Batty implemented the int4 QMoE emitter/packer at `751645b`; Chew approved with a caveat for one Ruff syntax regression under fix.
- Deckard produced rigorous Qwen-7B decode no-go results and a roofline verdict: the remaining int4 GEMVs are shared-memory/weight-read-efficiency limited, not grid-limited.
- Coordinator landed the generic profiling skill on `origin/main` as `e255985`.
- Stale-main cleanup note: `origin/main` still carries five old inbox notes (`marsten-glm4-static-split`, `marsten-phi-stacked-rebench`, `sebastian-qmoe-64expert`, `sebastian-qmoe-test-fix`, `sebastian-static-split-test`) that should be reconciled or removed in a future docs/state commit; no commit to `main` was made in this session.
