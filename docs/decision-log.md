# Decision Log

Each entry records what was decided, why, and what was given up. Written at the
time of the decision, not reconstructed afterwards.

---

## D-001 — Rebuild rather than patch the original repo

**Date:** Phase 0
**Decision:** Start a new repository (`pulseiq`) rather than refactor
`Real-Time-Competitor-Strategy-Tracker-for-E-Commerce` in place.

**Why:** The original is a single 400-line `scarpe.py` mixing scraping, model
inference, LLM calls, Slack notifications and Streamlit UI, with credentials
hardcoded. Nothing in it is testable without a browser and a live API key. The
git history also contains the leaked credentials permanently.

**Trade-off:** Loses the original commit history. Accepted — the original
history is a liability, not an asset.

---

## D-002 — Hardcoded credentials found and remediated

**Date:** Phase 0
**Finding:** `scarpe.py` lines 190–191 contained a live Groq API key and a
Slack webhook URL in plaintext, in a public repository.

**Action taken:**
1. Credentials treated as compromised and rotated.
2. All configuration moved to `.env` via `config/settings.py`
   (pydantic-settings). Secrets typed as `SecretStr`, so they render as
   `**********` in logs, tracebacks and `repr()`.
3. `.env` gitignored from commit #1; `.env.example` committed as the template.
4. `detect-secrets` wired into pre-commit and CI, with a committed baseline.

**Note for interviews:** removing the line from the file does *not* revoke the
credential. Rotation at the provider is the remediation; the code change only
prevents recurrence.

---

## D-003 — Two datastores: MongoDB (raw) + SQL (cleaned)

**Date:** Phase 1
**Decision:** Raw scrape documents to MongoDB Atlas; validated, typed rows to
SQLite (local) / Postgres (deployed).

**Why:** Raw scrape payloads are schemaless and nested, and their shape changes
whenever a target site changes its markup. Forcing them into columns discards
information needed to debug a failed run. Cleaned rows need constraints,
indexes and joins, which document stores handle poorly.

Each Mongo document carries `source`, `run_id` and `ingested_at`, which makes
"show me everything from the run that broke" a real query.

**Trade-off:** Two systems to operate. Mitigated by graceful degradation —
with `MONGODB_URI` unset the pipeline runs on SQLite alone.

---

## D-004 — Forecasting target derived from data, not fabricated

**Date:** Phase 2
**Decision:** Replace the original `Predicted_Discount` formula with a target
computed from observed prices.

**Why:** The original invented its target with an arithmetic formula, then
"predicted" it. Any model fitted to a deterministic function of its own inputs
scores near-perfectly and means nothing.

**Two targets now exist:**
- `discount_from_reference` — `(1 − price / trailing_max_price) × 100` over a
  *prior-only* window. Works on any price series, including scraped data with
  no discount field.
- Ground-truth `discount_pct` from Open Prices (`price_is_discounted` /
  `price_without_discount`), captured from receipts and shelf tags.

The second is preferred where available; the first makes the task definable
identically across sources.

---

## D-005 — Strictly chronological splitting

**Date:** Phase 2
**Decision:** All train/test splits are time-ordered and per-product.
`Split.assert_no_leakage()` verifies every training timestamp precedes every
test timestamp, and is called in tests and at the start of each training run.

**Why:** A random split on time series trains on Thursday and tests on
Wednesday. Error collapses and the reported metric is a lie that looks
excellent. Splitting per product additionally prevents a late-arriving product
landing entirely in test with no training history.

**Related:** every rolling/expanding feature is `.shift(1)`-ed before
`.rolling()`. Measured on a random-walk price series, the unshifted rolling
mean correlates 0.95 with the value being predicted versus 0.86 shifted — that
gap is free accuracy from a feature that contains the answer.

---

## D-006 — Open Prices (ODbL) as the price-history dataset

**Date:** Phase 2
**Decision:** Use the Open Prices Parquet export from Open Food Facts as the
primary training dataset for forecasting.

**Why:**
- Scraping cannot produce history retroactively. One run = one snapshot, so a
  scraper started today yields zero training rows today.
- Commercial price-history APIs (e.g. Keepa, ~€29/month plus API tokens)
  prohibit redistribution, which would make the results irreproducible for
  anyone reading the repo.
- Open Prices carries an explicit ground-truth discount label, which almost no
  free price dataset does.

**Licence — ODbL 1.0.** Two obligations, both handled in code:
1. *Attribution* — `ATTRIBUTION` constant in `seed_open_prices.py`, every row
   tagged `source="open_prices"`, credited in the README.
2. *Share-alike* — combining ODbL data with another database obliges releasing
   the combined result as open data. Open Prices rows are therefore never
   merged with scraped rows; they coexist in one table but stay separable by
   `source`, and exports filter on it. The provenance column is the mechanism
   that keeps share-alike contained.

**Trade-off:** Grocery products, not consumer electronics, and crowdsourced so
coverage is uneven. Accepted: the forecasting task is price-and-discount over
time, which is domain-independent, and the licence is unambiguous.

---

## D-007 — Amazon Reviews'23 for sentiment, non-commercial use only

**Date:** Phase 3
**Decision:** Use `McAuley-Lab/Amazon-Reviews-2023` for sentiment fine-tuning.

**Licensing position — stated plainly because it is not unambiguous.** The
dataset is governed by Amazon's Customer Reviews Terms of Use alongside an
MIT-style licence on the distribution repository; a derived copy on Hugging
Face is published as CC BY-SA 4.0. This is the standard corpus in the
recommender/NLP research literature.

**Use here is non-commercial portfolio and research only**, with citation:

> Hou, Y., Li, J., He, Z., Yan, A., Chen, X., McAuley, J. (2024).
> *Bridging Language and Items for Retrieval and Recommendation.*
> arXiv:2403.03952

**This dataset would need to be replaced if the project ever became
commercial.** Recorded here so that constraint is not forgotten.

**Known limitation — labels are a proxy.** Sentiment is derived from the star
rating (1–2 negative, 3 neutral, 4–5 positive). A 5-star review can still
contain complaints, so the ceiling on achievable accuracy is set by label
noise, not by the model. Reported metrics must be read with that in mind.

---

## D-008 — Selenium scraping restricted to permissive targets

**Date:** Phase 1
**Decision:** The live scraper targets `books.toscrape.com`, a sandbox site
published for scraping practice. Amazon selectors exist in `targets.yaml` but
are disabled (`products: []`).

**Why:** Amazon's Terms of Service prohibit scraping, their anti-bot measures
actively fight a Selenium session, and their DOM changes without notice. The
engineering being demonstrated — retry with backoff, rate limiting, config-driven
selectors, pure parsing separated from browser I/O — is demonstrated equally
well against a permissive target, without the legal ambiguity or the
maintenance treadmill.

`robots.txt` is to be checked and recorded here before any target is added.

---

## D-009 — Ruff runs from the venv, not a pinned pre-commit mirror

**Date:** Phase 2
**Decision:** `.pre-commit-config.yaml` uses `repo: local` with
`language: system` for ruff.

**Why:** The pinned hook revision (`v0.6.9`) drifted from the ruff installed by
`requirements.txt`. The hook passed locally while CI failed on `UP042` — the
local check was actively lying. One binary means hook and CI cannot disagree.

**Trade-off:** ruff must be installed in the environment. It is, via
`requirements.txt`.

---

## D-010 — Single `requirements.txt`; heavy ML deps deferred

**Date:** Phase 2
**Decision:** Consolidate the layered requirements files into one. Keep
`torch`/`transformers`/`peft` commented out until Phase 3.

**Why:** The default torch wheel bundles CUDA (~2.5 GB), useless on a laptop
and enough to exhaust a CI runner's disk. The CPU build is installed explicitly
at Phase 3.

Pinned `plotly<6` — Evidently requires it, and leaving it open caused repeated
up/downgrade churn.

---

## D-011 — AWS deferred to the end; design kept migration-ready

**Date:** Phase 2
**Decision:** Build and run entirely locally through Phase 6. Deploy to AWS
only at Phase 7.

**Constraints adopted now so that migration is configuration, not a rewrite:**
- File paths go through `fsspec`, so `./data/x.parquet` and
  `s3://bucket/x.parquet` are the same call.
- No local disk writes in the request path (Lambda's filesystem is read-only
  outside `/tmp`).
- Logging to stdout only, never files (CloudWatch captures stdout).
- Every external resource behind `settings` (`DATABASE_URL`,
  `MLFLOW_TRACKING_URI`), so SQLite → Aurora is a `.env` edit.
- No in-process schedulers or background threads (EventBridge triggers jobs).
- Stateless API; shared state lives in Redis.

**Target shape (all usage-billed, nothing always-on):** Lambda container +
Function URL, Aurora Serverless v2 at min 0 ACU with auto-pause, S3 for
artifacts, EventBridge for scheduling.

**Explicitly avoided:** NAT Gateway (~$32/month to merely exist — avoided by
using the Aurora Data API so Lambda needs no VPC), RDS Proxy (prevents
scale-to-zero), App Runner and ElastiCache (both bill while idle).

MongoDB Atlas M0, Upstash Redis, GitHub Actions and Streamlit Community Cloud
stay off AWS — their free tiers are permanent and the AWS equivalents cost more
for less.
