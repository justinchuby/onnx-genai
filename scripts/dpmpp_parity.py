"""Lock the DPM-Solver++ (2M) formula against diffusers before porting to Rust.

Drives diffusers `DPMSolverMultistepScheduler` (algorithm_type="dpmsolver++",
solver_order=2) and a hand-rolled Python port through an identical denoise loop
with a deterministic pseudo-model, and asserts the final latents match. Once this
passes, the same math is ported to onnx-genai's `Dpmpp2m` scheduler.

Run:  conda run -n onnx python scripts/dpmpp_parity.py
"""

import math

import numpy as np
import torch
from diffusers import DPMSolverMultistepScheduler

NUM_TRAIN = 1000
BETA_START = 0.00085
BETA_END = 0.012
NUM_STEPS = 12
SHAPE = (1, 4, 8, 8)


def sigmas_linspace(num_train, beta_start, beta_end, num_steps):
    """DPMSolverMultistep's inference sigma schedule (linspace spacing, zero last).

    Matches diffusers: N+1 linspace timesteps over [0, num_train-1], rounded to
    int, reversed, drop last; sigmas interpolated at those int timesteps; append 0.
    """
    denom = num_train - 1
    lo, hi = math.sqrt(beta_start), math.sqrt(beta_end)
    train = []
    prod = 1.0
    for i in range(num_train):
        beta = (lo + (hi - lo) * i / denom) ** 2
        prod *= 1.0 - beta
        train.append(math.sqrt((1.0 - prod) / prod))

    ts_full = [j * denom / num_steps for j in range(num_steps + 1)]  # linspace(0, N-1... ) N+1 pts
    ts_int = [int(round(t)) for t in ts_full][::-1][:-1]  # reverse, drop last -> N ints

    def interp(t):
        low = max(0, min(t, num_train - 1))
        return train[low]

    sig = [interp(t) for t in ts_int]
    sig.append(0.0)
    return sig


def alpha_sigma(sigma):
    alpha_t = 1.0 / math.sqrt(sigma * sigma + 1.0)
    sigma_t = sigma * alpha_t
    return alpha_t, sigma_t


def dpmpp_2m_port(sigmas, model, x0_seed):
    """Hand-rolled DPM++ 2M (dpmsolver++, midpoint, lower_order_final)."""
    num_steps = len(sigmas) - 1
    x = x0_seed.clone()
    prev_x0 = None
    for step in range(num_steps):
        sigma = sigmas[step]
        # scale_model_input is identity for DPMSolverMultistep.
        eps = model(x)
        alpha_t, sigma_t = alpha_sigma(sigma)
        x0 = (x - sigma_t * eps) / alpha_t  # convert_model_output (epsilon)

        lower_order_final = (step == num_steps - 1) and (num_steps < 15)
        order1 = (step == 0) or lower_order_final

        s_next = sigmas[step + 1]
        a_t, sig_t = alpha_sigma(s_next)
        a_s0, sig_s0 = alpha_sigma(sigma)
        lam_t = math.log(a_t) - math.log(sig_t) if sig_t > 0 else float("inf")
        lam_s0 = math.log(a_s0) - math.log(sig_s0)
        h = lam_t - lam_s0

        if order1 or prev_x0 is None:
            # first-order (DPM++1)
            x = (sig_t / sig_s0) * x - (a_t * (math.exp(-h) - 1.0)) * x0
        else:
            s_prev = sigmas[step - 1]
            a_s1, sig_s1 = alpha_sigma(s_prev)
            lam_s1 = math.log(a_s1) - math.log(sig_s1)
            h0 = lam_s0 - lam_s1
            r0 = h0 / h
            D0 = x0
            D1 = (1.0 / r0) * (x0 - prev_x0)
            x = (
                (sig_t / sig_s0) * x
                - (a_t * (math.exp(-h) - 1.0)) * D0
                - 0.5 * (a_t * (math.exp(-h) - 1.0)) * D1
            )
        prev_x0 = x0
    return x


def pseudo_model(x):
    return torch.tanh(x) * 0.5 + 0.1


def main():
    torch.manual_seed(0)
    noise = torch.randn(*SHAPE, dtype=torch.float64)

    ref = DPMSolverMultistepScheduler(
        num_train_timesteps=NUM_TRAIN,
        beta_start=BETA_START,
        beta_end=BETA_END,
        beta_schedule="scaled_linear",
        solver_order=2,
        algorithm_type="dpmsolver++",
        solver_type="midpoint",
        use_karras_sigmas=False,
        timestep_spacing="linspace",
        prediction_type="epsilon",
        lower_order_final=True,
    )
    ref.set_timesteps(NUM_STEPS)
    ref_lat = noise.clone() * ref.init_noise_sigma
    for t in ref.timesteps:
        scaled = ref.scale_model_input(ref_lat, t)
        eps = pseudo_model(scaled)
        ref_lat = ref.step(eps, t, ref_lat).prev_sample

    sig = sigmas_linspace(NUM_TRAIN, BETA_START, BETA_END, NUM_STEPS)
    my_lat = dpmpp_2m_port(sig, pseudo_model, noise.clone() * float(ref.init_noise_sigma))

    diff = float((ref_lat - my_lat).abs().max())
    print(f"diffusers init_noise_sigma={float(ref.init_noise_sigma):.4f} ours sigmas[0]={sig[0]:.4f}")
    print(f"max |Δlatent| (diffusers DPM++2M vs port) = {diff:.3e}")
    assert diff < 1e-4, f"DPM++ 2M mismatch: {diff}"
    print("\nPARITY OK: DPM++ 2M port matches diffusers DPMSolverMultistepScheduler")


if __name__ == "__main__":
    main()
