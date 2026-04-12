# Security Check (`security_check/`)

Threat model, security audit, and incident log for the crypto trading bot.

## Files

| File | Purpose |
|------|---------|
| `SECURITY.md` | Full threat model: API key handling, dependency risks, data exfiltration, etc. Numbered entries — append new findings as N+1. |

## Key rules

1. **`.env` is never committed.** API keys for Kraken (and any other exchange) live there.
2. **Kraken API keys are trade-only.** No withdrawal permissions. If a key is compromised, attacker can lose money but cannot exfiltrate funds.
3. **Sanitize all error logs.** Never log API keys or sensitive credentials, even on error paths.
4. **No third-party libraries with native code** added without review (supply chain risk).
5. **`requirements.txt` is pinned** for reproducibility and to detect tampering.

## When to update SECURITY.md

- Found a new vulnerability → add as next numbered entry
- Fixed a vulnerability → update the entry's status
- Added a new external dependency → assess and document
- Started using a new exchange/API → add threat model section
