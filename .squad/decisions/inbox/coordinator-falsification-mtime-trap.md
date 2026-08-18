### 2025-06: Restoring a falsification probe with Copy-Item/Move-Item silently defeats cargo

**By:** Coordinator

**What:** When you break a fix to falsify its test, do not restore the file with `Copy-Item`/`Move-Item` alone. Those cmdlets preserve the *original* `LastWriteTime`, so the restored file looks older than the artifact cargo built from the broken version. cargo's freshness check is mtime-based, sees nothing newer, and reuses the stale binary. Always follow a restore with:

```powershell
(Get-Item <path>).LastWriteTime = Get-Date
```

**Why:** This bit me on PR #1214. I falsified `structured_mode_is_withdrawn_when_the_message_token_does_not_resolve` by forcing `is_parseable` to `true` -- correctly RED. I restored the file, re-ran, and got 119 passed / **2 failed**. The source on disk was demonstrably correct; I read the exact line back and it said `self.message_token_id.is_some()`. For a few minutes the evidence said my fix did not work.

The dangerous version of this is the mirror image. Had I falsified in the other direction -- broken the code, restored, and seen **green** -- I would have concluded the test was worthless and deleted it, when in fact I was reading a stale binary of the *good* build. A falsification loop is precisely the moment when you are deliberately toggling one line and trusting the pass/fail signal completely, which is also precisely the moment a stale-build hazard does the most damage.

Cheap tell: if a result contradicts the source you just read, suspect the build before you suspect the code.
