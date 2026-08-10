# Decision: Delivery cadence — push early, draft PR early, ready when validated

**Date:** 2026-08-10T23:07:07.230+00:00
**By:** @justinchuby (via Copilot coordinator)
**Status:** Accepted — standing directive

## Decision

1. **Push progress to `origin` frequently.** Do not accumulate local-only
   commits; a completed increment belongs on the remote.
2. **Open draft PR(s) early**, as soon as a branch has meaningful content —
   rather than waiting for the workstream to be complete.
3. **Convert draft -> ready for review** once implementation *and* validation
   are complete.

## Why

Early pushes and draft PRs make in-flight work visible and recoverable. This
session demonstrated the failure mode directly: a lost coordinator session
left the entire EP plugin-export workstream uncommitted in the working tree,
and the recovered branch then accumulated 8 local-only commits that were
invisible from GitHub because push credentials were unavailable.

## Consequences

- The coordinator pushes after each milestone commit rather than only at the end.
- A draft PR is opened at the first milestone, then updated as work lands.
- Blocked pushes must be reported immediately and loudly, not deferred to the
  final summary, because they silently break this contract.
