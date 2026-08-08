### 2026-08-08: Server owns device memory authorities
**By:** Copilot
**What:** Multi-model server registries own one concurrency-safe device authority per backend/device compatibility domain and inject it into eager, lazy, admin, single-model, directory, config, and pipeline construction. Engine host/disk ledgers remain private; standalone engine constructors retain unique device authorities.
**Why:** A process/device physical-handle pool can only retain and transfer pages honestly when every production engine on that device charges one stable authority and ledger. Server ownership avoids a process-global registry and keeps authorities alive across individual model unloads.
