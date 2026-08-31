### 2026-08-26: Runtime shape reads depend on optional operand presence
**By:** Holden
**What:** Device-plugin pre-claim filtering classifies optional input-valued shape rules by whether the controlling input is present. Opset-13+ Squeeze uses a runtime input rule when axes is present and an explicit all-unit-dim rule when absent; opset-18 reductions decline only for present axes.
**Why:** Presence is the semantic property: Squeeze with device axes cannot be shaped safely on the host, while ReduceSum(X) reads no axes value and must remain claimable. Explicit empty axes also differ from absent axes for both Squeeze and `noop_with_empty_axes`, so the type variants preserve that distinction.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
