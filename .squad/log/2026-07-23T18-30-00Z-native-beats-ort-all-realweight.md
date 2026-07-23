# 2026-07-23T18:30:00Z — Native CUDA beats ORT on all real-weight models

- Deckard's on-device LongRoPE select merged as `97c1a56`; it removes Phi's host `If` seam through a conservative constant-branch `If`→`Where` lowering.
- Marsten's authoritative scoreboard merged as `f6bcde1`.
- Native CUDA now leads ORT for every available real-weight model: Qwen 0.5B +62.7%, 1.5B +36.8%, 7B +10.8%, and Phi +40.2%.
- The faster-than-ORT mandate is achieved; `deckard-6` test hardening remains in flight and is not logged as complete.
