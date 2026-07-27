### 2026-07-27: Keep inpainting conditioning outside scheduler state
**By:** Newt
**What:** Image diffusion carries only the 4-channel latent through the scheduler. The runner supplies a separate `{loop_endpoint}.conditioning` tensor containing `[mask | masked-image latent]`, and the engine appends it to form the 9-channel denoiser input each step.
**Why:** Schedulers must update only the noisy latent, while inpainting UNets require static 1+4 conditioning channels. This preserves the existing loop and final VAE decode contracts without checkpoint-specific dispatch.

### 2026-07-27: Zero strength means a zero-iteration tail
**By:** Newt
**What:** `start_step == num_steps` is valid and publishes the encoded seed directly to final pipeline phases.
**Why:** The documented `num_steps - round(num_steps * strength)` mapping produces exactly `num_steps` at strength 0.0; accepting it avoids an edge-case special case in front ends.
