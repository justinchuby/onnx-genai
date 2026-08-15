"""Lock the Karras sigma schedule against diffusers before porting to Rust.

Karras spacing (arXiv:2206.00364) is the most popular ComfyUI scheduler for
euler/dpmpp. onnx-genai's Euler/DPM++ math is already validated; Karras only
replaces the *sigma array*, so this checks that a from-scratch Karras sigma
computation matches diffusers' `EulerDiscreteScheduler` and
`DPMSolverMultistepScheduler` (both with use_karras_sigmas=True).

Run:  conda run -n onnx python scripts/karras_parity.py
"""

import math

import numpy as np
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler

NUM_TRAIN = 1000
BETA_START = 0.00085
BETA_END = 0.012
RHO = 7.0


def training_sigmas(num_train, beta_start, beta_end):
    denom = num_train - 1
    lo, hi = math.sqrt(beta_start), math.sqrt(beta_end)
    out = []
    prod = 1.0
    for i in range(num_train):
        beta = (lo + (hi - lo) * i / denom) ** 2
        prod *= 1.0 - beta
        out.append(math.sqrt((1.0 - prod) / prod))
    return out


def karras_sigmas(num_train, beta_start, beta_end, num_steps):
    """Karras rho=7 schedule from the training sigma range; append trailing 0."""
    train = training_sigmas(num_train, beta_start, beta_end)
    sigma_min = train[0]
    sigma_max = train[-1]
    min_inv = sigma_min ** (1.0 / RHO)
    max_inv = sigma_max ** (1.0 / RHO)
    ramp = [k / (num_steps - 1) for k in range(num_steps)] if num_steps > 1 else [0.0]
    sig = [(max_inv + r * (min_inv - max_inv)) ** RHO for r in ramp]
    sig.append(0.0)
    return sig


def exponential_sigmas(num_train, beta_start, beta_end, num_steps):
    """exp(linspace(log sigma_max, log sigma_min, n)); append trailing 0."""
    train = training_sigmas(num_train, beta_start, beta_end)
    log_min, log_max = math.log(train[0]), math.log(train[-1])
    ramp = [k / (num_steps - 1) for k in range(num_steps)] if num_steps > 1 else [0.0]
    sig = [math.exp(log_max + r * (log_min - log_max)) for r in ramp]
    sig.append(0.0)
    return sig


def check(name, ref_sigmas, num_steps, fn=karras_sigmas):
    mine = np.array(fn(NUM_TRAIN, BETA_START, BETA_END, num_steps))
    ref = np.array([float(s) for s in ref_sigmas])
    diff = float(np.abs(mine - ref).max())
    print(f"{name}: my sigmas[:3]={mine[:3]}  ref[:3]={ref[:3]}  max|Δ|={diff:.3e}")
    assert diff < 1e-4, f"{name} sigma mismatch: {diff}"


def main():
    for num_steps in (8, 20):
        e = EulerDiscreteScheduler(
            num_train_timesteps=NUM_TRAIN, beta_start=BETA_START, beta_end=BETA_END,
            beta_schedule="scaled_linear", use_karras_sigmas=True, timestep_spacing="linspace",
        )
        e.set_timesteps(num_steps)
        check(f"euler-karras N={num_steps}", e.sigmas, num_steps)

        d = DPMSolverMultistepScheduler(
            num_train_timesteps=NUM_TRAIN, beta_start=BETA_START, beta_end=BETA_END,
            beta_schedule="scaled_linear", algorithm_type="dpmsolver++", solver_order=2,
            use_karras_sigmas=True, timestep_spacing="linspace", final_sigmas_type="zero",
        )
        d.set_timesteps(num_steps)
        check(f"dpm++karras N={num_steps}", d.sigmas, num_steps)

        ex = EulerDiscreteScheduler(
            num_train_timesteps=NUM_TRAIN, beta_start=BETA_START, beta_end=BETA_END,
            beta_schedule="scaled_linear", use_exponential_sigmas=True, timestep_spacing="linspace",
        )
        ex.set_timesteps(num_steps)
        check(f"euler-exp    N={num_steps}", ex.sigmas, num_steps, exponential_sigmas)

    print("\nPARITY OK: Karras + exponential sigma schedules match diffusers")


if __name__ == "__main__":
    main()
