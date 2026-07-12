# RAI Policy

Authoritative check definitions for Rai. 🔴 Critical checks cannot be disabled.

## 🔴 Critical (cannot disable)
- Hardcoded credentials, API keys, tokens, secrets.
- Injection vulnerabilities (command, path, deserialization of untrusted input).
- Harmful or dangerous content generation.

## 🟡 Advisory (can disable with logged justification)
- PII exposure in logs or error messages.
- Bias indicators in defaults or heuristics.
- Missing rate limiting on public endpoints.
- Exclusionary or non-inclusive terminology.

## Terminology standards
- Prefer inclusive, neutral technical language.

## Opt-out
- Advisory checks may be temporarily disabled (auto re-enables after 30 days). Log to audit trail.
