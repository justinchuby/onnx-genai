### 2026-08-26T15-43-36: Plugin ABI tests must model ORT residency hooks; raw device types convert generically
**By:** Roy
**What:** Plugin ABI tests must model ORT residency hooks; raw device types convert generically
**References:** PR #2200 regression, branch fix/plugin-device-residency-abi
**Why:** WHAT: Keep #2200's mandatory GetTensorMemoryInfo/MemoryInfoGetDeviceType/Name/Id residency queries. Repair plugin_export_abi's synthetic OrtApi by supplying CPU memory-info callbacks rather than adding a blanket CPU fallback. Convert the platform-dependent raw device type through a generic checked u32 conversion, covering Linux u32 and Windows i32 typedefs. WHY: Real ORT API 27 supplies these hooks; the seven failures came from an incomplete mock, not an exported ABI layout/signature change. Falling back to CPU when hooks are absent would reopen device-pointer host-dereference risk.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
