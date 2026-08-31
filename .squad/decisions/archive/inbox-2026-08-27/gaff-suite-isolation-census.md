### 2026-08-26: CUDA suite-isolation policy has an independent census
**By:** Gaff
**What:** The source guard's policy set is exactly MHA, DFT, STFT, and NMS, independent from the target-to-lock mapping. It rejects an empty/mismatched mapping, missing or zero-test sources, missing first-statement acquisitions, immediately dropped guards, and wrong lock names.
**Why:** Using the implementation map as both inventory and expectation lets deletion erase the obligation it should enforce. NMS is included because it already intentionally uses the same target-local mutex policy.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
