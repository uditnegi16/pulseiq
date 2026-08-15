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

## Phase 3 — Sentiment Fine-Tuning ✅

**Delivered**
- Dataset loader: streaming, star→sentiment labels, class balancing,
  Parquet mirror fallback across sources
- Zero-shot baseline (`distilbert-base-uncased-finetuned-sst-2-english`)
- LoRA fine-tuning (DistilBERT, r=16, attention projections)
- Classification metrics with confusion matrix, macro-F1, majority-class floor,
  and error-reduction reporting
- Local CPU inference from the trained adapter
- Colab notebook driver — thin, calls tested modules

**Measured — test set, n=750**

| metric | zero-shot | fine-tuned | delta |
|---|---|---|---|
| accuracy | 0.8880 | **0.9413** | +0.0533 |
| precision | 0.9505 | 0.9485 | −0.0020 |
| recall | 0.8187 | **0.9333** | **+0.1147** |
| f1 | 0.8797 | **0.9409** | +0.0612 |
| macro_f1 | 0.8875 | **0.9413** | +0.0539 |

**Error reduction: 47.6%.**

| | |
|---|---|
| trainable parameters | 887,042 / 67,842,052 (**1.31%**) |
| adapter size | **4.27 MB** |
| training time | **61 seconds**, 3 epochs, Colab T4 |
| cost | **£0** |

**The finding, not just the number.** The improvement is almost entirely recall:
+11.5 points, against −0.2 on precision. The zero-shot model was *missing
negative reviews* — SST-2 was tuned on movie criticism, where negativity is
florid; product complaints are flat and factual, and it read too many as
neutral-positive. Fine-tuning corrected that without trading away precision.

Visible only because precision and recall were reported separately. An accuracy
figure alone would have said "+5 points" and hidden the mechanism.

**Ceiling.** Labels are star-derived proxies (D-013). At 94.1% on labels perhaps
~95% faithful, this is close to saturated; further gains would likely be fitting
label noise. Reported as a limitation rather than pursued.

---

## Phase 4 — Evaluation Gate ✅

**Delivered**
- `evaluation/thresholds.yaml` — versioned metric floors, with the recorded
  values they derive from and a stated tolerance
- `evaluation/regression_gate.py` — compares fresh reports against thresholds,
  fails the build on regression
- Reference reports committed under `reports/`, so CI has something to compare against
- Wired into GitHub Actions as a required step

**What it enforces**

| check | rule |
|---|---|
| `sentiment.accuracy` | >= 0.91 (recorded 0.9413) |
| `sentiment.macro_f1` | >= 0.91 (recorded 0.9413) |
| `sentiment.recall` | >= 0.90 — tighter floor, see below |
| `sentiment.*_vs_baseline` | must still beat zero-shot 0.8880 |
| `forecast.mae_h1..h12` | <= 0.01 / 0.02 / 0.05 / 0.07 |
| `forecast.leakage_tripwire` | no model may beat naive by >25% |

**Two design choices worth defending**

*Recall gets a tighter floor than accuracy.* The fine-tune's entire value was
recall (+11.5 points, precision flat). A change that trades recall back for
precision would leave accuracy stable while undoing the actual result — so
accuracy alone is not a sufficient guard.

*The gate also fails on results that are too good.* Every leakage bug in this
project's history made metrics look better, never worse (E-003 in particular). A
downside-only gate would have caught none of them. On near-random-walk price
data, a model suddenly beating naive by 40% is more likely leakage than skill,
so >25% improvement fails the build with "verify the split before trusting this".

*Missing reports SKIP rather than fail.* A PR touching only the scraper is not
blocked because nobody re-ran the fine-tune. `--strict` inverts this for release
builds, where "we never measured it" is not an acceptable answer.

**Verified against seven scenarios**, including a regressed model, a fine-tune
that stops beating its baseline, exploded forecast error, an implausible 80%
improvement, malformed JSON, and a zero-valued baseline (division-by-zero guard —
naive_last really does score exactly 0.0000 at h=1).

---

## Phase 5 — Serving Layer ✅

**Delivered**
- FastAPI service: `/health`, `/forecast`, `/forecast/products`,
  `/forecast/models`, `/sentiment`, `/recommend`, with OpenAPI docs at `/docs`
- Redis cache with a bounded in-memory fallback
- Multi-provider LLM routing: Groq primary, NVIDIA NIM fallback
- Streamlit dashboard consuming the API over HTTP
- `scripts/run_local.ps1` to start both processes

**Design decisions that carry weight**

*The dashboard consumes the API; it does not import models.* The model loads
once in one process rather than once per Streamlit session, the UI cannot
accidentally depend on training internals, and the same interface serves a UI, a
script or a cron job unchanged.

*Every forecast response carries `baseline_note`.* The measured finding is that
no model beats the naive forecast on this data. An API returning an ARIMA number
without that caveat would imply a precision the evaluation does not support, so
the honesty is in the payload rather than only in the docs.

*Degradation is graded, not binary.* `/health` returns `degraded` when Mongo,
Redis, the LLM or the sentiment adapter are unavailable, because forecasting
still works without any of them. A hard failure would pull the service out of a
load balancer for something it can operate without.

*`/recommend` returns 503 rather than canned text* when no LLM is configured. An
endpoint that silently returns filler is worse than one that fails: the caller
cannot distinguish analysis from placeholder.

*Cache failures are misses, not errors.* Every Redis operation swallows
connection errors and logs them. A cache is an optimisation; propagating a
timeout as a 500 would make it a liability.

**Verified**

| behaviour | result |
|---|---|
| missing adapter | `/sentiment` → 503, not 500 |
| unknown product | `/forecast` → 404 with a pointer to `/forecast/products` |
| invalid horizon | 422 |
| unknown model | 400, listing valid options |
| no LLM key | `/recommend` → 503 naming the variable to set |
| repeat request | `cached: false` then `cached: true` |
| cache keys | vary by product, model and horizon |
| trending series | naive / drift / ARIMA return genuinely different forecasts |
| API key leakage | provider errors never contain the key (asserted) |

---

## Phase 6 — Integration Testing ✅

**Delivered**
- `tests/integration/test_pipeline.py` — Parquet → validate → SQL → resample →
  split → models → metrics, on real components
- `tests/integration/test_api.py` — discovery → forecast → cache → error paths
  against a real database and a real app instance
- `tests/conftest.py` — shared `isolated_settings` fixture
- Separate CI step so a failure names the layer

**Why these exist separately.** Every unit test in this project passes, and
several real bugs still reached working code. They shared a shape: the pieces
were correct and the *joins* between them were not.

| bug | what the unit tests missed |
|---|---|
| E-002 | fixture used floats; the real Parquet used `Decimal`, and `float / Decimal` raises |
| E-003 | `split_per_product` produced splits its own leakage check rejected |
| E-013 | a test asserted a 503 that occurred only when the model happened to be absent |

A unit test with a fake on both sides of a boundary cannot detect a mismatch at
that boundary. These use real SQLite, real Parquet with `decimal128` columns,
and the real FastAPI app.

**Properties asserted**

- Decimal → float normalisation survives four modules (E-002 regression guard)
- Re-ingestion is idempotent: `inserted=N` then `inserted=0, skipped=N`
- `1,299.50` parses as 1299.50, not 129950 (E-001 regression guard)
- Splits pass `assert_no_leakage()` on data that came through the database
- Imputed rows are excluded from scoring — verified by comparing scored row
  counts with and without the filter
- Forecast error grows with horizon (the sanity property of any correct evaluation)
- Product identifiers round-trip exactly (`3001234567890@101` must survive so a
  client can feed the name back in)
- The cache returns identical *content*, not merely a `cached: true` flag
- Cache keys are scoped per model and per horizon
- `baseline_note` appears in the published OpenAPI schema, so the measured
  caveat is part of the contract rather than a droppable detail
- The gate reads the file names and metric keys the training code actually
  writes — a silent mismatch would make it skip every check and report success
  forever
- A fresh clone with an empty database produces no 500s

**Test counts:** 476 unit, 26 integration, 502 total. `pytest -m "not integration"`
for the fast inner loop.

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
| Regression gate | 24 |
| API, cache, LLM router | 59 |
| *(unit subtotal)* | *476* |
| Integration — pipeline | 12 |
| Integration — API | 14 |
| **Total** | **502** |

All green locally and in CI on fresh Ubuntu. No external services required —
the suite runs against in-memory SQLite, HTML fixtures, and injected fakes.

---

## Remaining

| Phase | Work | Estimate |
|---|---|---|
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
