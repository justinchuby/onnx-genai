"""Numeric parity: onnx-genai's EulerSchedule vs diffusers EulerDiscreteScheduler.

Replicates the exact Rust formulas (crates/onnx-genai-engine/src/pipeline.rs
`EulerSchedule`) in Python and drives both schedulers through an identical
multi-step denoise loop with a fixed pseudo-model, then asserts the final latents
match to floating-point tolerance.

Run:  conda run -n onnx python scripts/euler_parity.py
"""

import math

import numpy as np
import torch
from diffusers import EulerDiscreteScheduler

NUM_TRAIN = 1000
BETA_START = 0.00085
BETA_END = 0.012
NUM_STEPS = 5
SHAPE = (1, 4, 8, 8)


def rust_sigmas(num_train, beta_start, beta_end, num_steps):
    """Port of EulerSchedule::with_schedule (scaled_linear)."""
    denom = num_train - 1
    lo, hi = math.sqrt(beta_start), math.sqrt(beta_end)
    train_sigmas = []
    prod = 1.0
    for i in range(num_train):
        beta = (lo + (hi - lo) * i / denom) ** 2
        prod *= 1.0 - beta
        train_sigmas.append(math.sqrt((1.0 - prod) / prod))
    ts_denom = (num_steps - 1) if num_steps > 1 else 1

    def interp(t):
        low = max(0, int(math.floor(t)))
        high = min(low + 1, num_train - 1)
        frac = t - low
        return train_sigmas[low] * (1.0 - frac) + train_sigmas[high] * frac

    sigmas = []
    for k in range(num_steps):
        idx = num_steps - 1 - k
        t = idx * denom / ts_denom
        sigmas.append(interp(t))
    sigmas.append(0.0)
    return sigmas


def pseudo_model(scaled):
    # Deterministic stand-in for a denoiser noise prediction.
    return torch.tanh(scaled) * 0.5 + 0.1


def main():
    torch.manual_seed(0)
    noise = torch.randn(*SHAPE, dtype=torch.float64)

    # --- diffusers reference ---
    ref = EulerDiscreteScheduler(
        num_train_timesteps=NUM_TRAIN,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule="scaled_linear",
        timestep_spacing="linspace",
        interpolation_type="linear",
        prediction_type="epsilon",
    )
    ref.set_timesteps(NUM_STEPS)
    ref_lat = noise.clone() * ref.init_noise_sigma
    for i, t in enumerate(ref.timesteps):
        scaled = ref.scale_model_input(ref_lat, t)
        eps = pseudo_model(scaled)
        ref_lat = ref.step(eps, t, ref_lat).prev_sample

    # --- onnx-genai port ---
    sigmas = rust_sigmas(NUM_TRAIN, BETA_START, BETA_END, NUM_STEPS)
    my_lat = noise.clone() * sigmas[0]  # init_noise_sigma == sigmas[0]
    for step in range(NUM_STEPS):
        factor = math.sqrt(sigmas[step] ** 2 + 1.0)
        scaled = my_lat / factor
        eps = pseudo_model(scaled)
        dt = sigmas[step + 1] - sigmas[step]
        my_lat = my_lat + eps * dt

    ref_sigmas = ref.sigmas.to(torch.float64).numpy()
    my_sigmas = np.array(sigmas)
    sig_diff = float(np.abs(ref_sigmas - my_sigmas).max())
    lat_diff = float((ref_lat - my_lat).abs().max())

    print(f"diffusers sigmas: {ref_sigmas}")
    print(f"onnx-genai sigmas: {my_sigmas}")
    print(f"init_noise_sigma  ref={float(ref.init_noise_sigma):.6f}  ours={sigmas[0]:.6f}")
    print(f"max |Δsigma|   = {sig_diff:.3e}")
    print(f"max |Δlatent|  = {lat_diff:.3e}")

    # diffusers computes alphas_cumprod in float32; this port uses float64, so a
    # ~1e-5 relative rounding gap on sigmas (~O(10)) is expected and harmless.
    assert sig_diff < 1e-4, f"sigma schedule mismatch: {sig_diff}"
    assert lat_diff < 1e-4, f"latent mismatch: {lat_diff}"
    print("\nPARITY OK: onnx-genai Euler matches diffusers EulerDiscreteScheduler")


if __name__ == "__main__":
    main()
