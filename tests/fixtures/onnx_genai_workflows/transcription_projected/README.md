# Declared generative transcript projection

A synthetic Native head wraps its waveform/prompt-dependent token with `tok16`
and `tok17`. Both are registered special tokens; the response template
explicitly preserves them for request-local parsing. The version-1 transcription config supplies the exact decoder
prompt, language selections and optional context prompt; its declared content
region removes those framing tokens. Tests compare buffered, streamed and CLI
transcripts, actual token accounting, and conditioning-dependent inference.
No runtime code recognizes model-specific start/transcribe token spellings.
