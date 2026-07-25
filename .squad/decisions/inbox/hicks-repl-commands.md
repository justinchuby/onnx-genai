### 2026-07-24: REPL multimodal slash-command grammar
**By:** hicks
**What:** Added pure parsing for `/help`, `/reset`, `/raw`, `/system [text]`, `/image <path> [prompt text]`, and `/audio <path> [prompt text]`. Attachments stage for the next text turn, while single-line attachment commands immediately send their text. Missing paths warn without crashing; Phase 1 reports staged modalities and sends text only.
**Why:** This makes multimodal REPL input testable and extensible while honestly deferring engine-side image and audio execution to Phase 2.
