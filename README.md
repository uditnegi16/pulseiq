# PulseIQ - Competitor Pricing & Sentiment Intelligence Engine

Rebuild of a single-file Streamlit script into an engineered ML system:
time-aware discount forecasting, LoRA fine-tuned review sentiment,
experiment tracking, an evaluation gate in CI, and drift monitoring.

> **Status:** Phase 0 - scaffolding.

## Quickstart (Windows / PowerShell)
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements\base.txt -r requirements\dev.txt
pip install -e .
Copy-Item .env.example .env    # then fill in your own keys
```

## Layout
| Path | Purpose |
|---|---|
| `src/pulseiq/ingestion/` | Selenium scraping, parsing, validation |
| `src/pulseiq/storage/` | MongoDB (raw) + SQL (cleaned) |
| `src/pulseiq/training/` | Forecasting + sentiment fine-tuning |
| `src/pulseiq/evaluation/` | Metrics, eval harness, CI regression gate |
| `src/pulseiq/monitoring/` | Evidently drift reports |
| `src/pulseiq/api/` | FastAPI serving layer |
| `app/` | Streamlit dashboard (consumes the API) |

## Results
See `docs/metrics.md` - no numbers claimed until they are measured.

## Security note
This project began from a codebase with hardcoded credentials.
See `docs/security.md` for the remediation writeup.
