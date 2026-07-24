# GLM-5.2 IndexShare and ORT benchmark fix

GLM-5.2 DSA-MoE now decodes native CUDA end-to-end in eager mode: Mobius emits
`pkg.nxrt::IndexShare` and the runtime supplies its shape handler. The Mobius
half is folded into PR #404 for Justin to merge; the runtime half is on main.
The ORT benchmark backend now defaults CUDA graphs off, making its eager
baseline reliable. Follow-up: GLM-5.2 DSA CUDA-graph capture remains in
progress because symbolic auxiliary output axes prevent static capture.
