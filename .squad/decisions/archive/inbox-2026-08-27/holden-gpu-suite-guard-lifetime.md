### 2026-08-26: CUDA suite-lock source guard proves guard lifetime
**By:** Holden
**What:** PR #2184's source guard accepts only a first-statement named lock guard whose identifier is never used again in the masked test body. Any later exact identifier use fails closed, covering explicit/qualified drop, assignment, shadowing, and moves; comments and strings remain masked.
**Why:** Merely checking acquisition allowed `let guard = lock(); drop(guard); run();` to pass while all GPU work ran unprotected. Rust locals otherwise live through their enclosing function, so first-statement binding plus no later use is a scoped, inspectable lifetime proof without inventing a full Rust parser.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
