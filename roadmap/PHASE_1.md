# Phase 1: Project Setup & Data Pipeline

## Goal
Establish a clean project structure, version control, dependency management, and refactor the existing data scraper into a reusable module.

## What This Phase Builds On
Nothing — this is the foundation.

## Implementation Steps

1. **Initialize git repo**
   - `git init`, create `.gitignore` (exclude `.env`, `__pycache__/`, `*.csv`, `.DS_Store`)
   - CSV data files are large and regenerable — keep them out of git

2. **Set up Python project structure**
   ```
   crypto/
   ├── src/
   │   ├── data/
   │   │   ├── __init__.py
   │   │   ├── scraper.py        # refactored from final_log_scraper.py
   │   │   ├── volume_profile.py # VP computation logic
   │   │   └── validator.py      # data quality checks
   │   ├── features/             # Phase 2
   │   ├── models/               # Phase 3
   │   ├── trading/              # Phase 4-5
   │   └── config.py             # centralized configuration
   ├── data/                     # local data directory (gitignored)
   │   └── raw/                  # raw OHLCV CSVs
   ├── tests/
   ├── requirements.txt
   └── .env                      # API keys (gitignored)
   ```

3. **Create requirements.txt**
   - ccxt, pandas, numpy, torch, python-dotenv
   - Pin versions for reproducibility

4. **Refactor `final_log_scraper.py`**
   - Extract configuration into `src/config.py`
   - Extract data fetching into `src/data/scraper.py`
   - Extract VP computation into `src/data/volume_profile.py`
   - Add a CLI entry point: `python -m src.data.scraper` to regenerate data

5. **Add data validation** (`src/data/validator.py`)
   - Check for gaps in timestamps (missing candles)
   - Check for zero/negative prices or volumes
   - Check VP rows sum to ~1.0 (if normalized)
   - Print a summary report

6. **Environment variable setup**
   - `.env` file for API keys (Kraken keys needed later, but structure now)
   - `python-dotenv` to load them

## Test When Done
- [ ] `git status` shows clean repo with proper `.gitignore`
- [ ] `pip install -r requirements.txt` works in a fresh venv
- [ ] `python -m src.data.scraper` regenerates the CSV from scratch
- [ ] `python -m src.data.validator` reports data quality on existing CSV
- [ ] No API keys or CSV files in git history
