# Rosenthal — History


## 2026-07-28T05-49-08+0000 — Wave 3 update
Approved PR #321, including the critical CUDA stream-ordering check: host-blocking H2D is the producer and completes before kernel enqueue, so no race is introduced versus resident upload.
