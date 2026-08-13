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

---

## D-012 — Review data sourced from Parquet mirrors, not the canonical repo

**Date:** Phase 3
**Decision:** Load Amazon Reviews'23 through Parquet-backed republications
rather than `McAuley-Lab/Amazon-Reviews-2023` directly.

**Why:** `datasets` 4.0 removed `trust_remote_code` and no longer executes
dataset loading scripts. The canonical repo still serves its `raw_review_*`
configs through a script (only `raw_meta_*` were converted to Parquet), so the
review data is unreachable through `load_dataset` there regardless of arguments.

**Implementation:** `open_review_stream()` holds an ordered list of sources and
falls through on failure, so one dead mirror does not block a run. Total failure
raises listing every attempt plus the root cause.

**Trade-off, stated plainly:** these are third-party copies. The data should be
byte-identical to the original, but the provenance is one step removed and the
mirrors could disappear. The alternative — pinning `datasets<4.0` — trades a
supply-chain risk for a dependency-rot risk, and would eventually conflict with
`transformers`.

**If both mirrors vanish:** pin `datasets==3.6.0` in a dedicated environment, or
download the category archives directly from
`amazon-reviews-2023.github.io` and load from local Parquet.

---

## D-013 — Sentiment labels derived from star ratings, with a stated ceiling

**Date:** Phase 3
**Decision:** Binary labels from star ratings — 1-2 negative, 4-5 positive,
3-star reviews dropped entirely.

**Why drop 3 stars:** it is where the star-to-sentiment mapping is least
trustworthy. A 3-star review is genuinely mixed text; forcing it into a class
teaches the model that mixed text has a definite label, and keeping it as a
third class means training on the noisiest examples in the set.

**Why balance the classes:** roughly 80% of Amazon reviews are 4-5 stars. On the
raw distribution, a model that always predicts "positive" scores 80% and appears
to work. Balancing makes accuracy an honest metric rather than a restatement of
the base rate. `majority_baseline()` is reported alongside every result so the
do-nothing floor is always visible.

**The limitation that must accompany any reported number:** these are *proxy*
labels. People leave 5 stars with complaints in the text, and 1 star because
delivery was late on a product they liked. The achievable accuracy ceiling is
therefore set by label noise, not model capacity. A fine-tune reaching ~92%
where labels are ~95% faithful has saturated the task; pushing further would be
fitting the noise.

---

## D-014 — Three-way split; the test set is scored exactly once

**Date:** Phase 3
**Decision:** train / validation / test at 70/15/15. Checkpoint selection uses
validation macro-F1. The test set is touched once, at the end.

**Why:** selecting a checkpoint on test performance is the classification
equivalent of the temporal leakage this project fixed in forecasting (D-005). It
produces a number that is optimistic by an unknown amount and cannot be
reproduced on new data.

A random split is correct here, unlike the forecasting task — reviews are
independent observations with no temporal ordering to respect.

**Enforced, not just intended:** the Trainer receives `eval_dataset=val_ds` and
never sees the test frame. A test asserts this.

**Splits are persisted to parquet** so the zero-shot baseline and the fine-tuned
model are scored on byte-identical test data. Regenerating the split between
runs — even with the same seed but a different sklearn version — would
invalidate the before/after comparison.

---

## D-015 — LoRA rather than full fine-tuning

**Date:** Phase 3
**Decision:** Freeze DistilBERT and train LoRA adapters on the attention
projections (`q_lin`, `v_lin`), r=16, alpha=32.

**Why:**
1. The adapter is a few MB and can be committed to git. A full checkpoint is
   ~250MB, which makes the result unreproducible for anyone cloning the repo.
2. Training fits on a free-tier T4.
3. The base model is untouched, so the before/after comparison isolates domain
   adaptation rather than confounding it with catastrophic forgetting.

**Trade-off:** a small accuracy ceiling relative to full fine-tuning. On a binary
task with proxy labels that ceiling is set by label noise long before it is set
by LoRA capacity (D-013), so the trade is close to free here.

**Baseline choice matters as much as the method.** The zero-shot comparator is
`distilbert-base-uncased-finetuned-sst-2-english` — already sentiment-tuned,
just on movie reviews. Both models are DistilBERT, so the measured delta
isolates *domain adaptation*. A weaker baseline would have produced a larger and
less meaningful improvement number.

---

## D-016 — Training compute on Colab, inference local

**Date:** Phase 3
**Decision:** Fine-tune on a Colab T4 via the official VS Code extension; run
inference locally on CPU.

**Why:** DistilBERT + LoRA on 20k reviews is ~10 minutes per epoch on a T4
versus 2-4 hours on a laptop CPU. Fine-tuning needs several runs before the
numbers are worth reporting, and one-shot experimentation is not experimentation.

**Structure this forces, and it is an improvement:** all logic lives in tested
modules under `src/`; the notebook clones the repo and calls them. Nothing
substantive lives in notebook cells. The Colab runtime is a separate machine, so
the notebook installs dependencies, clones, and downloads the adapter at the end.

**Cost: free.** No Colab Pro, no AWS credits spent.

---

## D-017 — Metric thresholds committed, and the gate fails on good news too

**Date:** Phase 4
**Decision:** `thresholds.yaml` holds versioned metric floors enforced by CI.
The gate fails a build both when a metric regresses **and** when an improvement
is implausibly large.

**Why the upside check exists:** every leakage bug in this project made metrics
look *better*, never worse. E-003 (a global date comparison on a per-product
split) and the shifted-vs-unshifted rolling mean (0.86 vs 0.95 correlation with
the target) both improved apparent accuracy. A gate checking only the downside
would have caught neither. On near-random-walk price data, a model beating naive
by more than 25% is more likely a broken split than a breakthrough.

**Why recall has a tighter floor than accuracy:** the fine-tune's measured value
was entirely recall (+11.5 points, precision −0.2). A change trading recall back
for precision keeps accuracy flat while undoing the result. Guarding the headline
metric alone would not guard the finding.

**Why missing reports skip rather than fail:** blocking a scraper PR because
nobody re-ran a fine-tune trains people to bypass the gate. `--strict` exists for
release builds where the stricter rule is appropriate.

**Governance:** thresholds are raised in the same PR as the improvement that
justifies them, with the measured value in the commit message. They are lowered
only with a written justification here. Lowering a threshold to make a build pass
is how a gate becomes decoration.
