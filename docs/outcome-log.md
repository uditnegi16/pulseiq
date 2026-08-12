# Outcome Log

What was actually built, what it measured, and what remains. Updated as each
phase closes. Companion to `decision-log.md` (why choices were made) and
`error-log.md` (what broke and why).

Nothing here is projected or aspirational. If a number appears, it was measured.

---

## Phase 0 — Foundations ✅

**Delivered**
- Repository structure: `ingestion/`, `storage/`, `features/`, `training/`,
  `evaluation/`, `monitoring/`, `api/`, `app/`, `tests/`, `docs/`
- `config/settings.py` — every credential typed and loaded from `.env`;
  secrets are `SecretStr` so they render as `**********` in logs and tracebacks
- `.env` gitignored from commit #1; `.env.example` committed as the template
- Nine pre-commit hooks including `detect-secrets` with a committed baseline
- GitHub Actions CI

**Security finding**
The inherited codebase carried a live Groq API key and a Slack webhook in
plaintext at `scarpe.py:190-191`, in a public repository. Treated as compromised
and rotated. The distinction that matters: *deleting the line does not revoke the
credential* — rotation at the provider is the remediation, the code change only
prevents recurrence.

**Verified:** a commit containing a fake key is rejected by the hook.

---

## Phase 1 — Ingestion & Storage ✅

**Delivered**
- Hardened Selenium scraper: driver lifecycle as a context manager, retry with
  exponential backoff and jitter, rate limiting, selectors as configuration
- Parsing split from browser I/O, so parse logic is testable offline
- Validation layer producing a `ValidationReport` — rows are never silently
  dropped; every rejection is counted by reason
- MongoDB Atlas (raw documents) + SQLite/Postgres (cleaned rows)
- Ingestion CLI with `--dry-run`, `--from-csv`, `--open-prices`
- `storage/healthcheck.py` — verifies connectivity before a long run, with
  credentials masked in its output

**Measured**

| Metric | Value |
|---|---|
| Rows ingested (Open Prices, ODbL) | **6,341** cleaned to SQL |
| Raw documents in MongoDB Atlas | **7,073** |
| Validation pass rate | **89.7%** (rejections: 732 same-day duplicates) |
| Re-run result | `inserted=0 skipped_existing=6341` |

That last row is the point: ingestion is **idempotent**, enforced by a
`UNIQUE(product_name, observed_on)` constraint at the database level rather than
only in application code. A scheduled daily scrape is safe to retry.

**Bugs fixed from the original**
1. Driver leak — `get_driver()` never called `quit()`, so every exception leaked
   a Chrome process
2. Retry-everything — bare `except Exception` with a fixed 5s sleep meant a typo
   in a selector retried three times before failing
3. No rate limiting — products were hit back to back
4. `extract_price` stripped the decimal point, turning `1,299.50` into `129950`
5. Parse failure returned `0`, making a free item indistinguishable from an error

---

## Phase 2 — Forecasting ✅

**Delivered**
- Monthly resampling with imputation tracking (`is_imputed` per row)
- Strictly chronological, per-product splits with `assert_no_leakage()`
- Trailing-only feature engineering — every rolling window `.shift(1)`-ed
- Five baselines + ARIMA (AIC order selection) + Prophet
- Evaluation harness with paired comparison and fallback-rate reporting
- Horizon-curve evaluation at h = 1, 3, 6, 12 using expanding origins
- MLflow tracking (SQLite backend), runs pinned to git commits

**Measured — Run C, 56 series, 3,430 evaluations**

Median MAE by forecast horizon (months ahead):

| model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| **naive_last** | **0.0000** | **0.0100** | **0.0333** | **0.0418** |
| arima_auto | 0.0087 | 0.0205 | 0.0428 | 0.0623 |
| moving_average_3 | 0.0117 | 0.0300 | 0.0433 | 0.0633 |
| prophet¹ | — | — | — | — |
| mean | 0.1557 | 0.1727 | 0.2073 | 0.3300 |

¹ Excluded from Run C for iteration speed; placed last in Run B (MAE 0.1742).

Paired win rate against naive: **9–29% at every horizon.** No model exceeds 29%.

**Headline finding — reported, not buried**

The naive forecast is unbeaten, and the reason is one number: **median MAE at
h=1 is exactly 0.0000.** More than half the series do not change price month to
month. Where the price is unchanged, "predict the last value" is not a heuristic
— it is exactly correct, and there is no error left to remove.

Three checks that this is the correct result rather than a broken evaluation:
1. For a random walk the naive forecast is provably optimal, and retail prices
   are near-random-walk step functions. A large win would suggest leakage.
2. Error grows monotonically with horizon for every model.
3. Simple methods beat complex ones on short, noisy, level-shifting series —
   which is what should happen.

**Fallback rate: 0% for every model.** No result is a silently degraded naive
forecast wearing another model's name.

**What this redirects.** The defensible claim is not "I beat the baseline". It is
that price forecasting here is solved at short horizons, so modelling effort
belongs on *when* a price changes, not *what* it changes to. That points at
discount detection, where the remaining signal is.

---

## Phase 3 — Sentiment Fine-Tuning 🔄

**Delivered**
- Dataset loader: streaming, star→sentiment labels, class balancing,
  Parquet mirror fallback
- Zero-shot baseline (`distilbert-base-uncased-finetuned-sst-2-english`)
- LoRA fine-tuning module (DistilBERT, r=16, attention projections)
- Classification metrics with confusion matrix, macro-F1, majority-class floor,
  and error-reduction reporting
- Local CPU inference from the trained adapter
- Colab notebook driver (22 cells) — thin, calls tested modules

**Not yet measured.** The fine-tuning run has not completed. Numbers go here and
in `docs/metrics.md` when it does; nothing is claimed until then.

**Ceiling to report alongside whatever the result is:** labels are star-derived
proxies (D-013). Accuracy is bounded by label noise, not model capacity.

---

## Test coverage

| Suite | Tests |
|---|---|
| Parsing | 57 |
| Features & splits | 45 |
| Forecasting baselines | 55 |
| Forecasting models & harness | 30 |
| Horizon curve | 15 |
| Open Prices loader | 39 |
| Sentiment & classification | 59 |
| Storage & validation | 63 |
| Scraper | 26 |
| **Total** | **389** |

All green locally and in CI on fresh Ubuntu. No external services required —
the suite runs against in-memory SQLite, HTML fixtures, and injected fakes.

---

## Remaining

| Phase | Work | Estimate |
|---|---|---|
| 3 | Run the fine-tune, record before/after | 1h |
| 4 | Eval harness, `regression_gate.py`, CI metric gate | 2h |
| 5 | FastAPI + Streamlit + Redis cache + LLM routing | 4–5h |
| 6 | Integration tests | 1.5h |
| 7 | Evidently drift monitoring, Docker, deployment | 3h |
| — | `architecture.md`, `security.md`, `phase-log.md`, README | 2h |
| — | AWS migration (deferred by design, D-011) | 3–4h |

---

## Credentials status

| Variable | Needed for | Status |
|---|---|---|
| `MONGODB_URI` | Raw document store | ✅ Live (Atlas M0, free) |
| `MLFLOW_TRACKING_URI` | Experiment tracking | ✅ SQLite, no credential |
| `DATABASE_URL` | Cleaned rows | ✅ SQLite, no credential |
| `GROQ_API_KEY` | LLM recommendations | ⬜ Phase 5 |
| `REDIS_URL` | Caching | ⬜ Phase 5, optional |
| `SLACK_WEBHOOK_URL` | Notifications | ⬜ Phase 5, optional |
| `NVIDIA_NIM_API_KEY` | LLM fallback | ⬜ Phase 5, optional |

Phases 0–4 run on **zero paid services**. Total spend to date: **£0**.
