### 2026-08-26T20-13-24: Construct CUDA page-in fence tests with explicit device/host handshakes
**By:** Deckard
**What:** Construct CUDA page-in fence tests with explicit device/host handshakes
**References:** issue #1896, PR #2235, commit 383c3e548e6cccdd0c2186d61f5c6969472da581
**Why:** Issue #1896 was a test-apparatus defect, not a demonstrated production fence defect. Under locked parallel load, a tagged consumer proved the old negative arm read the prior payload because synchronous-H2D poison priming had not established poison. The replacement uses a drained compute-stream fill, a host-mapped release gate, and a D2H completion handshake; destructive wait bypass must fail the named test.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
