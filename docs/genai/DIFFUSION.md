# Diffusion workflows

Diffusion uses the same workflow IR as every other package. A typical graph invokes a
text/media encoder in setup, loops over a denoiser and an ONNX scheduler/solver policy
component, carries the latent state, and invokes a decoder/postprocessor before an
`emit` node.

The loop's zero-based induction value binds the solver `step` port. Schedule tensors,
RNG counter state, guidance inputs, and termination are ordinary typed SSA values.
Masked diffusion replaces the solver with a masked-update policy component and carries
both state and mask. No diffusion algorithm or family name is implemented in the host.

See [WORKFLOW_POLICY_COMPONENTS.md](../WORKFLOW_POLICY_COMPONENTS.md) for exact solver and
masked-update ONNX port contracts.
