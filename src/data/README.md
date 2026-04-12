# Data Module (`src/data/`)

Multi-exchange OHLCV scraping, volume profile computation, and funding rate ingestion.

## Files

| File | Purpose |
|------|---------|
| `__main__.py` | CLI entry point. Run `python -m src.data scrape [timeframe]` or `python -m src.data validate`. |
| `scraper.py` | Multi-exchange OHLCV fetcher. Currently fetches from Bitstamp + Coinbase via ccxt, then merges by averaging prices and summing volumes. |
| `volume_profile.py` | Computes the 50-bin relative volume profile (VP) per candle. For each row, looks back 180 days and bins log-returns weighted by volume. |
| `validator.py` | Sanity checks on the merged CSV: gaps, price anomalies, VP normalization. |
| `funding_rate.py` | Loads BTC perpetual funding rate from two sources: Binance CSV (2020-2023, downloaded from GitHub) + Gate.io API (2024-present). Forward-fills 8h rate to bar frequency. |

## Usage

### Scrape OHLCV
```bash
python -m src.data scrape          # 1h timeframe (default), → BTC_1h_RELVP.csv
python -m src.data scrape 15m      # 15min, → BTC_15m_RELVP.csv
python -m src.data scrape 5m       # 5min (slow, careful with rate limits)
```

The scraper reads `TIMEFRAME` from config.py by default; passing an arg overrides it via deferred imports (so the data CLI can scrape any timeframe without editing config).

### Validate existing CSV
```bash
python -m src.data validate                  # validates the default CSV path
python -m src.data validate path/to/file.csv # specific file
```

### Use funding rate in code
```python
from src.data.funding_rate import load_funding_rate
fr = load_funding_rate(ohlcv_dates=df['date'])
# Returns DataFrame with [date, funding_rate, funding_zscore, funding_trend]
```

## Output

Files are written to the project root (NOT inside src/):
- `BTC_<timeframe>_RELVP.csv` — features
- `BTC_<timeframe>_RELVP_metadata.json` — provenance info
- `data/funding_rate_merged.csv` — merged funding rate cache

## Design notes

- **Bitstamp + Coinbase only**: chosen for historical depth and US-friendly access. To add more exchanges, edit `EXCHANGES` in `config.py`.
- **VP is timeframe-agnostic**: the same code works for 1h, 15m, 5m, etc. Only the bar count changes.
- **Funding rate is from external sources**: Binance API is geo-blocked from the user's location, so historical CSVs are pre-downloaded. OKX and Gate.io APIs are used for live data.
