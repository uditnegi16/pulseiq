# Error Log

Bugs that reached working code, what caused them, and what changed so the class
of bug cannot recur. Written when they happened, not reconstructed.

A pattern runs through most of these: **a test fixture built from an assumption
matches the assumption, not reality.**

---

## E-001 — Rolling-mean regex parsed `"1299"` as `129`

**Phase:** 1 · **Caught by:** unit tests, first run

`_NUMBER = r"\d{1,3}(?:,\d{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?"` — the comma group
used `*`, so on an ungrouped number the first alternative matched greedily short
and returned the first three digits.

**Fix:** require at least one comma group (`+`) in the grouped branch so plain
numbers fall through to the second alternative.

**Why it was caught:** the tests were written alongside the code with real price
strings (`₹24,999.00`, `1299`, `INR 45999.00`) rather than after the fact.

---

## E-002 — Parquet DECIMAL columns crashed the Open Prices loader

**Phase:** 2 · **Caught by:** first run against real data · **Severity:** blocking

```
TypeError: unsupported operand type(s) for /: 'float' and 'decimal.Decimal'
```

Open Prices stores monetary columns as Parquet `DECIMAL`, which pandas surfaces
as `decimal.Decimal`. `float / Decimal` raises. **34 tests passed** because the
fixture used Python floats — it was built from the *column names* in the
documentation without checking the *types*.

**Fix:** `_as_float()` normalises Decimal / float / NaN / None / str at every
point a price enters arithmetic. The fixture now emits `Decimal` by default,
with `test_decimal_and_float_paths_agree` asserting the two produce identical
results.

**Lesson:** round-trip through the real file format at least once before
trusting a fixture. Documentation gives you names, not dtypes.

---

## E-003 — `split_per_product` produced splits its own leakage check rejected

**Phase:** 2 · **Caught by:** realistic simulation, not the unit tests

`Split.assert_no_leakage()` compared `train.max()` against `test.min()`
**globally**. Across products with different lifespans, product A's last
training date is naturally later than product B's first test date — valid, not
leakage. The guard raised on correct partitions.

The original fixtures gave every product an identical date range, so it never
fired. A simulation with staggered spans hit it immediately.

**Fix:** the check is group-aware — per-product when a group column exists,
global otherwise. Four regression tests use staggered ranges, including one
confirming genuine within-product leakage is still caught.

**Lesson:** uniform fixtures hide bugs that only appear under heterogeneity.
Simulate the shape of the real data, not a tidy version of it.

---

## E-004 — Pre-commit ruff and CI ruff were different versions

**Phase:** 2 · **Caught by:** CI · **Severity:** trust

`.pre-commit-config.yaml` pinned `ruff-pre-commit v0.6.9`; `requirements.txt`
installed the latest. Two different linters with different rule sets. The hook
reported **Passed** locally while CI failed on `UP042`.

The worst kind of failure: the local check was actively lying.

**Fix:** ruff runs via `repo: local` with `language: system`, using the ruff in
the venv — the same binary CI uses. Hook and CI cannot disagree.

**Lesson:** any tool pinned in two places will drift. Pin it once.

---

## E-005 — MLflow 3.15 removed the filesystem tracking store

**Phase:** 2 · **Caught by:** first tracked run

```
MlflowException: The filesystem tracking backend (e.g., './mlruns') is in
maintenance mode ... set MLFLOW_ALLOW_FILE_STORE=true to opt out
```

`MLFLOW_TRACKING_URI=file:./mlruns` — a widely-copied default — now raises
outright.

**Fix:** default changed to `sqlite:///mlflow.db`, which is the recommended
local backend and the one that migrates cleanly to Postgres + S3 in Phase 7.

**Lesson:** defaults copied from tutorials age badly. Verify against the
installed version, not the blog post.

---

## E-006 — Prophet silently degraded to a naive forecast

**Phase:** 2 · **Caught by:** unit test · **Severity:** would have corrupted results

Prophet performs bare-integer timedelta arithmetic internally, which numpy
>= 2.5 deprecates. Because this project promotes `DeprecationWarning` to an
error under pytest, Prophet's `fit()` raised — and the adapter caught it and
fell back to predicting the last value.

Without the `used_fallback` flag, the leaderboard would have reported a Prophet
MAE that was actually the naive baseline's, and the model comparison would have
been quietly meaningless.

**Fix:** scoped `@pytest.mark.filterwarnings("ignore::DeprecationWarning")` to
the Prophet test class only — the global strict rule stays, since that is what
caught E-003. Added a monkeypatched test asserting a failed fit is always
recorded in `used_fallback` and logged at WARNING, and the harness leaderboard
now reports `fallback_rate` per model.

**Lesson:** every graceful fallback must be observable. A fallback that is not
counted is indistinguishable from success, and in an ML comparison that means
crediting one model with another's score.

---

## Recurring themes

1. **Fixtures encode assumptions.** E-002 and E-003 both passed their tests and
   failed on real data, because the fixture matched what was believed rather
   than what existed.
2. **Silent degradation is worse than failure.** E-006 would have produced
   plausible, wrong numbers. Visible fallbacks and a `fallback_rate` column are
   the countermeasure.
3. **Strict settings pay for themselves.** Promoting `DeprecationWarning` to an
   error surfaced E-006 immediately. Turning that rule off would have been
   easier and wrong; the ignore is scoped instead.
