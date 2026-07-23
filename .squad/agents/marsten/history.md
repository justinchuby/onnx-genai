# Marsten — History

- 2026-07-23: Re-benchmarked post-fusion CUDA performance, established the Qwen regression was real code rather than CPU noise, and confirmed `12efc92` restored Qwen 7B to 286.73 tok/s.
- 2026-07-23T15:45:00Z: Nsight/profile analysis found Phi's primary remaining gap is eager Greater and host If capture seams: 2.258 ms/token (43.8% of wall time) is outside 2.899 ms/token GPU work. Fixed-decode control-flow specialization is the priority over further GEMV tuning.
