# Raw Data (`data/raw/`)

Per-exchange raw OHLCV CSVs from the scraper.

## Status

**Empty in repo** — these files are gitignored. They're regenerated when you run:
```bash
python -m src.data scrape          # 1h
python -m src.data scrape 15m      # 15min
```

## What gets created here (after scraping)

| File | Source | Format |
|------|--------|--------|
| `bitstamp_<symbol>_<timeframe>.csv` | Bitstamp exchange OHLCV | timestamp, open, high, low, close, volume |
| `coinbase_<symbol>_<timeframe>.csv` | Coinbase exchange OHLCV | timestamp, open, high, low, close, volume |

## How they're used

`src/data/scraper.py` writes one CSV per exchange. `merge_exchanges()` then combines them by:
- Averaging prices across non-zero contributors
- Summing volumes
- Producing the merged dataset that becomes input to `compute_relative_vp()`

## Why gitignored

These files are large (50-200 MB at 1h, 200-800 MB at 15min) and easily regenerated from the source exchanges. Storing them in git would bloat the repo unnecessarily. The merged + VP-computed CSV at project root (`BTC_<timeframe>_RELVP.csv`) is what most code uses.
