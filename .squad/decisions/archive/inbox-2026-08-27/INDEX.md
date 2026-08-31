# Decision inbox consolidation — 2026-08-27

**By:** Scribe
**Source queue:** `.squad/decisions/inbox/`
**Canonical standing summary:** `.squad/decisions.md`

This archive preserves every source drop's line content and ordering. Checkout may
normalize line endings, and archiving inserted an EOF newline where a source lacked
one so that the one-line archive-provenance suffix starts on its own line. The source
wording was not rewritten. Class `A` means the durable decision was not previously
represented with enough fidelity and was merged into the live standing summary or
retained here as historical narrative. Class `C` means the drop was superseded or
corrected; the chain and final rule are recorded below. No entry remained pending
or ambiguous.

## Body integrity

To reproduce each SHA-256, normalize CRLF and lone CR bytes to LF; validate and
remove the final archive-provenance HTML comment line (including the comment's
optional terminal LF, but retaining the LF immediately before the comment); then
strip at most one terminal LF from the remaining body and hash those bytes. For a
pinned source, normalize its line endings and strip at most one terminal LF before
comparison. This intentionally places the presence or absence of one EOF newline
outside the normalized body hash while preserving every additional terminal LF, so
extra blank lines remain significant. `BASE-VERIFIED` means the resulting bodies
match the source tracked at `origin/main`
`7d183fe2f0daf635e025387c434fe3998dd4874e`. `UNVERIFIED current-body-only`
means the source was never tracked, so no historical preimage exists; the hash
anchors the current archived line content without claiming historical verification.

| Source entry | Verification | Normalized body SHA-256 |
|---|---|---|
| `Pris-unique-and-nms-remain-cuda-capture-unsupported-by-.md` | UNVERIFIED current-body-only | `c39d23be5606beae671b6b2997fd745c98dd8b58940f3127cd7345a9ac698cd0` |
| `Batty-device-plugins-conservatively-decline-host-value-r.md` | UNVERIFIED current-body-only | `6c9bf3608e483a7b7720484b3dff53445e40ccb330f9141752800b922bdc5708` |
| `chew-pr353-thin-m-gemm-review.md` | BASE-VERIFIED | `a376e0f447776f0298eb23b0c62c7de91913ea21978dd487f68d7a938f4327d7` |
| `chew-pr359-clip-fast-review.md` | BASE-VERIFIED | `73e361abe5fc9bfe2837b661bc2364e8143024be56fc57e7fabd485f1dcb0320` |
| `chew-pr361-neon-relu-review.md` | BASE-VERIFIED | `691f8bbf845ef592944505d3ea6b889c3158b82dd826dab69b2a5186f5a81606` |
| `chew-pr366-dense-elementwise-review.md` | BASE-VERIFIED | `8d89ea19a2c1b3f99245223ea7832c81f5cce5df67eb63974495dc35f96ee552` |
| `chew-pr368-dft-perch-review.md` | BASE-VERIFIED | `8ecf2d5a5e50fb6aead91777467ccf8e692eff45a515b42befc32c20494c1fdc` |
| `coordinator-criterion-failure-is-structural-not-probabilistic.md` | BASE-VERIFIED | `a968728b0c94bb636d6b82930855bd4b27ceaeb45631925b3d5c5f98effc0fe3` |
| `coordinator-in-environments-that-cannot-exercise-the-code-writ.md` | BASE-VERIFIED | `5adfc4a877d8442627fe9e8db9ba6bd5e330874e3c83aee88083930b5d4e266d` |
| `coordinator-mutation-testing-harnesses-fail-toward-false-confi.md` | BASE-VERIFIED | `bbcd4b2320f04df2a33cd5e30bc360ab1b1edd853d1a9d301f00b558b70bb12b` |
| `coordinator-mutation-testing-is-the-acceptance-bar-for-the-mem.md` | BASE-VERIFIED | `fc0d1e67424cddba5193e01fe49105292a6bc0073bff4f2ecece75aa23d56133` |
| `coordinator-test-defects-recurse-the-fix-for-a-level-n-test-de.md` | BASE-VERIFIED | `514395247c1cb4b7a483a04ae37d73cc8cf4b3ed803fc4f496a94669bbc8761f` |
| `copilot-coordinator-add-final-vmm-only-cuda-phase.md` | BASE-VERIFIED | `c509faf2565a373e23dad7bd5fbb72755d2cdfd3153850254ced773069522f7d` |
| `copilot-coordinator-keep-quartz-publishing-deliberately-simple.md` | BASE-VERIFIED | `70094c01f5fb9fb95497659f1e28ae0b468468b178862d1e8568dc652663ebdc` |
| `copilot-coordinator-separate-capability-discovery-from-release-safety.md` | BASE-VERIFIED | `77ed786b5aad6b0c62ae7749f72c4984682182db819e90451389127ff59b09dc` |
| `Copilot-keep-phase-1-memory-api-extraction-mechanism-only.md` | BASE-VERIFIED | `979f36d4967848d9760c29952fae59b627c705e630bc5d489da877a8fcf184b9` |
| `Copilot-phase-5-process-memory-manager-ownership-boundarie.md` | BASE-VERIFIED | `e8fb79401c0828e72d9e5225859626d7b3d883199a1ef61b44b46c60137799c3` |
| `copilot-slice7c-boundary-consumer-wiring.md` | BASE-VERIFIED | `02b56b6764511b8b555fec3f3193ddbcf4285389b9e1aff2a66412566494434e` |
| `Copilot-slice7d-route-residency-production-binding.md` | BASE-VERIFIED | `bbdb28ce6e9a763404b37e0f3cddad5ed6605081ac818cc927d0d5e8dab40293` |
| `deckard-1810-composable-vmm-spike-results.md` | BASE-VERIFIED | `6c5adc54d9b914d55e29127dac77685b2937eb391de6cf7bb9146f7264c8d378` |
| `deckard-1810-slice6-route-telemetry.md` | BASE-VERIFIED | `9d0f4a687f51e1f7a8d4ffb79d5ba43b120e44714e6275ac9fa24eb02aad85bb` |
| `deckard-a-prime-spike-results-cycle7.md` | BASE-VERIFIED | `5ea1b505d498ca15fe3aeacc7f1280911a3f4ad751897f3847bf1ad611e50b62` |
| `Deckard-construct-cuda-page-in-fence-tests-with-explicit-d.md` | UNVERIFIED current-body-only | `51142fd54e742c5471ea7c77f4518dcf276af403c03c04e30202ef2704f86238` |
| `deckard-cuda-mha-safety.md` | UNVERIFIED current-body-only | `381dd751cc8ee561c83b7b0f6e0637eeae5e0352f941a408aa708c4482545dcd` |
| `deckard-reshape-zerocopy-view.md` | BASE-VERIFIED | `57cd7580f4783b0bbd8e203d2447db0381a4613aa494441d4d86d47d9c25aa90` |
| `Freysa-serialize-resource-sensitive-cuda-integration-targ.md` | UNVERIFIED current-body-only | `713a17982f787318bd83278cb654a54e7cda773a834d1be86c2859f7b8abaf8d` |
| `gaff-cudnn-vmm-doc-truth.md` | UNVERIFIED current-body-only | `a82bee6ba213fdfba9c458c8745104be07a7ab67648f9b4259091d769a12c2fe` |
| `gaff-deepseek-v2-cuda-coherence.md` | BASE-VERIFIED | `6d3fd2873a81fff5d773f85ab777bdd64fa69b924d96c25407f6d7d677ba1a69` |
| `gaff-deepseek-v2-mask-graph-capture.md` | BASE-VERIFIED | `26de8ae39bb7a9215a5a71b14915d5f4eee252d7cfd6a2ac605543e69c0c64e3` |
| `gaff-mtp-verify-graph-capture.md` | BASE-VERIFIED | `9d6cfd77a30b631d4d30960d6d63eab82f52829aa9e4d2e24e33d84aa4f0b394` |
| `gaff-production-test-ci-honesty.md` | UNVERIFIED current-body-only | `087739c0fa8cd036b803010f6be1cc1afdeb6e9477dee4f9dcdfa3d3bf7f50f0` |
| `gaff-suite-isolation-census.md` | UNVERIFIED current-body-only | `10ac88e391da0a24a78da3fa3233a986304934f8862dea7e6cae4daf3aab6c31` |
| `holden-allocator-teardown.md` | UNVERIFIED current-body-only | `ea009e28b9d6606be75596e5e12543a13a22272f70fca2f2570980209e02606a` |
| `holden-gpu-suite-guard-lifetime.md` | UNVERIFIED current-body-only | `77c0c506abc453b9931e99a5dd6df43a3d8629b86f807bc5128d3ae7684c1c38` |
| `holden-plugin-shape-present-axes.md` | UNVERIFIED current-body-only | `b3715db4c7a33dd2a175bbd151058437d44bb890f692b33bc6ec50769226cb33` |
| `iran-dense-elementwise-simd.md` | BASE-VERIFIED | `acddc278305772b46826c39f3bf1b875b641a92c2c0c2f9f06c786a68f53ab56` |
| `Isidore-canonicalize-merge-gate-fixture-roots-and-fail-har.md` | UNVERIFIED current-body-only | `c97ef88ac9e86d83f9aff7426038b800029aa2a7327b4ca194695e649d88bcdb` |
| `Isidore-classify-onnx-fixture-paths-before-generic-docs-pa.md` | UNVERIFIED current-body-only | `848ed07bc6198c09e246bbd54b21202e66e31e8a5f73ba7a4aedfef9437e68ac` |
| `Isidore-pin-cuda-python-packaging-to-the-validated-13-1-ru.md` | UNVERIFIED current-body-only | `1337516f655dadda0c3a0f45bcf59748148a44207f32f7a3907a902454bdfe91` |
| `Leon-cuda-mha-indices-promote-before-the-first-multipli.md` | UNVERIFIED current-body-only | `1bb63e268cadf604f083a7404427014ac29ad9c5f872c1294c3a50c14a976ae2` |
| `Leon-cuda-nms-uses-bounded-deviceworkspace-selection-wi.md` | UNVERIFIED current-body-only | `7a49bdcfcfe8d31b141726f372d760972e07b214773ad5950cf59f8d06b1e2fe` |
| `leon-paged-attention-3a1-kv-index.md` | BASE-VERIFIED | `913cc699ea6f16aecf7fdf231fe23640d8f30a2dcf8c86ee9047974dd90fef0d` |
| `leon-paged-attention-audit-rev2.md` | BASE-VERIFIED | `bb1e92110d80254b20e127e16ede9e52f34a7a72aafbef5aabed82a748923f76` |
| `roy-1896-r4-wait-proof.md` | UNVERIFIED current-body-only | `c3b7223c2c3b19cda7b82332677dc57ec1b6ecee50cbc1abb03f6b9b36493643` |
| `Roy-graph-fixture-extension-policy-strips-trailing-asc.md` | UNVERIFIED current-body-only | `2f823c03a6192931918db75e1766ae998f5e6b1a1b00d24d750a73623ff85111` |
| `Roy-heterogeneous-static-execution-rejects-kernel-size.md` | UNVERIFIED current-body-only | `a059eee0cb354379080faea09a21ebb7d3a873f998d2a2163d4acff2e9da4822` |
| `Roy-plugin-abi-tests-must-model-ort-residency-hooks-ra.md` | UNVERIFIED current-body-only | `e520540c2658e280481eeec9bfa60c38eb519c55f1338c8faa478f3c8eccc22d` |
| `roy-session-kv-cache.md` | BASE-VERIFIED | `35467743dcdba4a2f7107b9c200cac6d2cbb618bac4ed803019651042e8663b9` |
| `Sapper-mobius-owns-explicit-batching-declarations-runtime.md` | UNVERIFIED current-body-only | `9d19958c86e2eecf2dbc13ac3438e8a5acf4dd9e306c85534013a8f9ca884b4a` |
| `sebastian-decode-budget-physical-cores.md` | BASE-VERIFIED | `34df77b018ed5b3617eed9ede6b46c49b2a57e032fcadb851f13994d24ce3f11` |
| `sebastian-small-model-ttft.md` | BASE-VERIFIED | `78b9122777c7c96f9b88d22ec0c67e4856e5155541ba69e94fefd7f3d0508f84` |
| `sebastian-steady-state-prefill.md` | BASE-VERIFIED | `ad73cda7137bf8e3a4d7f165f883aaf72b5d8b1cb72e8d9adf5a2a4a918d7005` |

## Inventory

| Entry | Class | Canonical disposition |
|---|:---:|---|
| `Batty-device-plugins-conservatively-decline-host-value-r.md` | A | Standing plugin-residency rule live; detail archived |
| `Copilot-keep-phase-1-memory-api-extraction-mechanism-only.md` | A | Final memory-stack boundaries live; phase detail archived |
| `Copilot-phase-5-process-memory-manager-ownership-boundarie.md` | A | Final memory-stack boundaries live; phase detail archived |
| `Copilot-slice7d-route-residency-production-binding.md` | A | Current route-residency status live; slice detail archived |
| `Deckard-construct-cuda-page-in-fence-tests-with-explicit-d.md` | A | Final #1896 rule live; apparatus detail archived |
| `Freysa-serialize-resource-sensitive-cuda-integration-targ.md` | A | Standing CUDA-suite isolation rule live |
| `Isidore-canonicalize-merge-gate-fixture-roots-and-fail-har.md` | C | Closed unmerged and superseded by #2227 deletion |
| `Isidore-classify-onnx-fixture-paths-before-generic-docs-pa.md` | A | Standing fixture-classification rule live |
| `Isidore-pin-cuda-python-packaging-to-the-validated-13-1-ru.md` | A | Standing CUDA packaging rule live |
| `Leon-cuda-mha-indices-promote-before-the-first-multipli.md` | A | Standing checked-geometry rule live |
| `Leon-cuda-nms-uses-bounded-deviceworkspace-selection-wi.md` | A | Standing CUDA dynamic-output rule live |
| `Pris-unique-and-nms-remain-cuda-capture-unsupported-by-.md` | A | Standing CUDA dynamic-output rule live; archived as `agent-Pris-...` to avoid the Windows `PR*.md` ignore collision |
| `Roy-graph-fixture-extension-policy-strips-trailing-asc.md` | A | Standing fixture-classification rule live |
| `Roy-heterogeneous-static-execution-rejects-kernel-size.md` | A | Standing heterogeneous-execution rule live |
| `Roy-plugin-abi-tests-must-model-ort-residency-hooks-ra.md` | A | Standing plugin ABI test rule live |
| `Sapper-mobius-owns-explicit-batching-declarations-runtime.md` | A | Standing producer-authored batching rule live |
| `chew-pr353-thin-m-gemm-review.md` | A | Historical merged review archived |
| `chew-pr359-clip-fast-review.md` | C | NaN rationale superseded by #361 experiment |
| `chew-pr361-neon-relu-review.md` | A | Final NaN/Relu ruling archived |
| `chew-pr366-dense-elementwise-review.md` | C | Rejection preserved; final PR #366 later merged |
| `chew-pr368-dft-perch-review.md` | A | Historical merged review archived |
| `coordinator-criterion-failure-is-structural-not-probabilistic.md` | A | Standing evidence-criterion rule live; full rationale archived |
| `coordinator-in-environments-that-cannot-exercise-the-code-writ.md` | A | Standing unavailable-environment evidence rule live |
| `coordinator-mutation-testing-harnesses-fail-toward-false-confi.md` | A | Standing mutation-harness rule live |
| `coordinator-mutation-testing-is-the-acceptance-bar-for-the-mem.md` | A | Standing memory-safety acceptance rule live |
| `coordinator-test-defects-recurse-the-fix-for-a-level-n-test-de.md` | A | Standing recursive-test validation rule live |
| `copilot-coordinator-add-final-vmm-only-cuda-phase.md` | A | Final memory-stack boundaries live; phase detail archived |
| `copilot-coordinator-keep-quartz-publishing-deliberately-simple.md` | A | Standing Quartz publishing rule live |
| `copilot-coordinator-separate-capability-discovery-from-release-safety.md` | A | Final memory-stack boundaries live; phase detail archived |
| `copilot-slice7c-boundary-consumer-wiring.md` | A | Current route-residency status live; slice detail archived |
| `deckard-1810-composable-vmm-spike-results.md` | A | Historical feasibility evidence archived; invariant live |
| `deckard-1810-slice6-route-telemetry.md` | A | Historical telemetry evidence archived; invariant live |
| `deckard-a-prime-spike-results-cycle7.md` | A | Historical corrected NO-GO archived |
| `deckard-cuda-mha-safety.md` | A | Standing checked-geometry/RAII rule live |
| `deckard-reshape-zerocopy-view.md` | A | Historical merged optimization archived |
| `gaff-cudnn-vmm-doc-truth.md` | A | Standing cuDNN claim/package rule live |
| `gaff-deepseek-v2-cuda-coherence.md` | A | Historical coherence fix archived |
| `gaff-deepseek-v2-mask-graph-capture.md` | A | Historical capture fix archived |
| `gaff-mtp-verify-graph-capture.md` | C | Wrong-logit blocker superseded; dual-slot crash fix retained |
| `gaff-production-test-ci-honesty.md` | A | Standing CI evidence rule live |
| `gaff-suite-isolation-census.md` | A | Standing CUDA-suite census rule live |
| `holden-allocator-teardown.md` | A | Standing plugin allocator ownership rule live |
| `holden-gpu-suite-guard-lifetime.md` | A | Standing CUDA lock-lifetime rule live |
| `holden-plugin-shape-present-axes.md` | A | Standing plugin optional-input rule live |
| `iran-dense-elementwise-simd.md` | C | Original pending proposal superseded by review/final merge |
| `leon-paged-attention-3a1-kv-index.md` | A | Paged-KV one-authority rule live; merged detail archived |
| `leon-paged-attention-audit-rev2.md` | A | Paged-KV one-authority rule live; merged detail archived |
| `roy-1896-r4-wait-proof.md` | C | Deterministic-POISON proof superseded by final layered proof |
| `roy-session-kv-cache.md` | C | Design superseded by #397 implementation/#408 benchmark correction |
| `sebastian-decode-budget-physical-cores.md` | A | Standing CPU decode-budget rule live |
| `sebastian-small-model-ttft.md` | C | Pre-#353 projection superseded by measured post-#353 state |
| `sebastian-steady-state-prefill.md` | A | Historical measured post-#353 state archived |

## Correction chains and status fixes

- `chew-pr359-clip-fast-review.md` inverted the NEON NaN mapping. The later
  `chew-pr361-neon-relu-review.md` direct experiment is authoritative:
  `vmaxq_f32`/`vminq_f32` propagate NaN; `vmaxnmq_f32`/`vminnmq_f32` suppress it.
- Iran's original #366 proposal and Chew's rejection remain as the review chain.
  The queue's `Pending`/`REJECT` states are historical: PR #366 subsequently
  merged as head `7995750b1a`.
- The earlier July archive calls the Perch DFT change PR #357. The reviewed and
  merged PR represented by `chew-pr368-dft-perch-review.md` is PR #368
  (`7679459d50` final head).
- Roy's native session-KV design was implemented by PR #397; PR #408 corrected
  the benchmark to use that persistent-session API. The archived design's
  `not implemented` state and projections are not current directives.
- `gaff-mtp-verify-graph-capture.md` correctly preserves the generic dual-slot
  graph-liveness fix that merged in PR #1690, but its assertion that MTP M>=2
  verify logits are structurally wrong is superseded by the live 2026-08-21
  finding: recurrent correctness can be token-identical and current MTP is
  blocked by step economics/throughput instead.
- The PagedAttention audit and index-emission drops say `NOT merged`; current
  status is merged in PRs #1940 and #1955 respectively.
- Slice 7C and 7D status text is historical. They merged in PRs #2007 and #2046.
  Slice 7E (#2082) and Slice 7F (#2163) remain open, so a live pointer remains.
- `Isidore-canonicalize-merge-gate-fixture-roots-and-fail-har.md` describes
  closed, unmerged PR #2223. The owner stopped that line as over-engineered, and
  PR #2227 deleted both optional merge-gate scripts. It is historical only.
- Roy's revision-4 #1896 schedule claimed deterministic POISON after deleting
  only the wait. That claim is superseded by the final #2235 layered proof:
  removing one ordering edge cannot create the reverse edge under the identical
  fixed schedule. Deckard's later explicit-handshake apparatus is retained as
  separate falsifiability evidence, not as a production-schedule equivalence.
