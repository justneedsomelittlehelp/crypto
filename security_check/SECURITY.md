# Security

> This project handles API keys and real money. Security failures have direct financial consequences.

---

## Architecture Security Overview

### API Key Management
- All API keys stored in `.env` file (gitignored, never committed)
- Loaded via `python-dotenv` at runtime
- Kraken API keys should be scoped: **trade permission only, no withdraw**
- No API keys in code, config files, logs, or error messages

### Attack Surface
| Surface | Risk | Mitigation |
|---------|------|------------|
| `.env` file | Key theft if machine is compromised | File permissions (600), separate trading account with limited funds |
| Kraken API | Unauthorized trades | Trade-only API keys, no withdraw |
| VPS | Remote access to bot | SSH key auth only, firewall, fail2ban |
| Git history | Accidentally committed secrets | `.gitignore` from day 1, pre-commit hook to scan for secrets |
| Log files | Keys/tokens in logs | Never log API keys; sanitize error messages |

### Operational Security
- Trading account should hold only capital allocated for bot trading — not savings
- Regularly rotate API keys (quarterly)
- Monitor Kraken account activity independently of the bot
- Have an emergency procedure: how to manually stop the bot and cancel all open orders

## Threat Model

| Threat | Likelihood | Impact | Mitigation |
|--------|-----------|--------|------------|
| API key leaked in git | Medium (before setup) | High — unauthorized trades | .gitignore, pre-commit hooks |
| VPS compromised | Low | High — key theft, unauthorized trades | SSH keys, firewall, limited API permissions |
| Bot bug places bad order | Medium | Medium — financial loss | Risk manager, position limits, paper trading |
| Kraken API returns bad data | Low | Medium — wrong trade decision | Data validation, sanity checks on prices |
| Model goes haywire | Medium | Medium — series of bad trades | Circuit breakers, daily loss limits |

## Resolved Vulnerabilities

(None yet — project is in setup phase)

## Outstanding Items

- [ ] Create `.gitignore` with `.env` exclusion before first commit
- [ ] Set up pre-commit hook to scan for secrets (e.g., `detect-secrets`)
- [ ] Configure Kraken API keys with trade-only permissions (no withdraw)
- [ ] Document emergency shutdown procedure
- [ ] Set up VPS with SSH key auth only (Phase 6)
