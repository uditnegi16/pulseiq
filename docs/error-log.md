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

---

## E-007 — `datasets` 4.0 removed `trust_remote_code`, breaking the review loader

**Phase:** 3 · **Caught by:** first Colab run · **Severity:** blocking

```
`trust_remote_code` is not supported anymore.
Please check that the Hugging Face dataset 'McAuley-Lab/Amazon-Reviews-2023'
isn't based on a loading script and remove `trust_remote_code`.
```

The parameter was written from the dataset's own documentation without checking
it against the installed library version. `datasets` 4.0 removed it entirely and
now refuses to execute dataset scripts at all.

Dropping the argument would not have helped: the `raw_review_*` configs are
still script-backed (only `raw_meta_*` were converted to Parquet), so the review
data is unreachable through `load_dataset` at the canonical repo regardless of
arguments.

**Fix:** `open_review_stream()` tries Parquet-backed mirrors in order and falls
through on failure. If every source fails it raises listing each attempt and the
root cause, rather than propagating whichever exception came last.

**Trade-off accepted:** the mirrors are third-party republications, not the
McAuley Lab original. Recorded in `docs/decision-log.md` D-012 — provenance
matters even when the bytes are identical.

**Lesson (third occurrence of this pattern):** verify library APIs against the
installed version, not against documentation or tutorials. See also E-002
(Parquet dtypes) and E-005 (MLflow file store).

---

## E-008 — Secret scanner flagged documentation placeholders

**Phase:** 3 · **Caught by:** GitHub secret scanning · **Severity:** false positive

Two alerts on commit `ddf319c`:
- `docs/mongodb-setup.md:60` — `pulseiq_app:YOUR_PASSWORD@pulseiq.xxxxx.mongodb.net`
- `tests/unit/test_storage.py:271` — `admin:hunter2@cluster0.abc.mongodb.net`

Both are placeholders. The second is a *test asserting that credentials get
masked* — the scanner matched the URI shape without evaluating the context.

**Verified before dismissing.** `git log --all -S "<real cluster id>"` returned
nothing, confirming the live connection string was never committed. Dismissing a
secret alert without that check would be exactly the wrong instinct.

**Fix:** fixtures now use RFC 2606 reserved names (`example.invalid`,
`EXAMPLE_USER`, `EXAMPLE_PASSWORD`), which can never resolve and read
unambiguously as non-real. Alerts closed as "used in tests" / "false positive".

**Lesson:** investigate every alert, then make the fixture obviously fake so it
cannot recur. Silencing the scanner would have been faster and wrong.

---

## E-009 — Commits silently lost to pre-commit hook aborts

**Phase:** 2-3 · **Caught by:** `git log` audit · **Severity:** process

An entire phase of work (7 files, ~1,900 lines) appeared committed but was not.
`end-of-file-fixer` and `ruff-format` modify files during the hook run, which
aborts the commit by design — the modified files need re-staging. `git push`
then reported `Everything up-to-date`, which reads like success.

Happened three times before being noticed.

**Fix (process, not code):** always finish with `git log --oneline -1` and
confirm the message is the intended one. `Everything up-to-date` means *nothing
was committed*, not that everything is fine.

**Lesson:** a tool that "fails" as part of working correctly will be misread as
success. The verification step has to be explicit.

---

---

## E-010 — Notebook variable shadowed by an imported function

**Phase:** 3 · **Caught by:** first fine-tuning run · **Severity:** blocking

```
TypeError: 'function' object is not subscriptable
```

The notebook bound `train` to the training DataFrame, then a later cell ran
`from ...finetune_lora import train`. The import silently rebound the name, so
`train(train, val)` passed the function as its own first argument.

Notebook-specific: in a module this would be caught by any linter, but notebook
cells share one namespace across the whole session and execute in whatever order
the user chooses.

**Fix:** import aliased to `run_training`, frames renamed `train_df` / `val_df` /
`test_df`, and the recovery cell rebuilds the splits from `frame` with the same
seed rather than depending on what `train` currently points at.

**Lesson:** in notebooks, never give a DataFrame the same name as anything
importable. `train`, `test`, `eval`, `input` and `next` are the usual offenders.

---

## E-011 — Colab's preinstalled torchao too old for peft

**Phase:** 3 · **Caught by:** first fine-tuning run · **Severity:** blocking

```
ImportError: Found an incompatible version of torchao.
Found version 0.10.0, but only versions above 0.16.0 are supported
```

peft's LoRA dispatcher calls `is_torchao_available()` unconditionally, and that
function *raises* on an old version rather than returning False. Colab ships
torchao 0.10.0 preinstalled. Nothing in this project uses torchao at all — it
was reached purely as part of peft's dispatcher chain.

**Fix:** upgrade or uninstall torchao, then restart the kernel. Uninstalling is
the lower-risk option: peft returns early when the package is absent, and
removing an unused dependency cannot conflict with torch's pins.

**Lesson:** a managed environment's preinstalled packages are part of your
dependency graph whether you asked for them or not. "Works locally" and "works
on Colab" are different claims.

---

---

## E-012 — A validator defined twice silently disabled itself

**Phase:** 5 · **Caught by:** a spurious Redis warning in test output

```
Redis init failed (Redis URL must specify one of the following schemes
(redis://, rediss://, unix://)) -- falling back to in-memory cache
```

`REDIS_URL=` was blank in `.env`, and the `blank_is_unset` validator in
`config/settings.py` listed `redis_url` among its fields — so it should have
become `None`. It did not, because a patch had been applied twice and the
validator was **defined twice with the same decorator**. Python keeps only the
last definition, and pydantic registered only that one, so the field list on the
first was silently discarded.

The symptom was misleading: the error named a URL scheme, which points at the
value, when the fault was in the validator that should have removed the value.
The fallback then worked correctly, which made it a warning rather than a
failure and easy to dismiss.

**Fix:** one definition, with a note in the docstring recording that duplication
disables it.

**Lesson:** duplicate definitions do not error in Python, they overwrite. A
patch applied twice can therefore leave code that reads correctly and behaves
as if it were absent. When a fix appears not to have taken effect, check whether
it took effect *twice*.

---

## E-013 — An environment-dependent test passed only where the model was absent

**Phase:** 5 · **Caught by:** the first run on a machine with the adapter present

`test_missing_adapter_returns_503_not_500` asserted that `/sentiment` returns
503. It passed in the sandbox where `models/sentiment_lora/` was empty, and
failed on the development machine, which had the trained adapter — the endpoint
correctly returned 200 with a real prediction.

The test asserted an environmental accident rather than a behaviour.

**Fix:** the failure is now forced with `monkeypatch` rather than assumed from
the filesystem, and a second test covers the success path — skipping, not
failing, when the adapter is absent, since it is a 4 MB artefact of a Colab run
rather than something the repository guarantees.

**Lesson:** a test that depends on what happens to be installed is testing the
machine, not the code. If a precondition matters, force it.

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
4. **Documentation ages faster than code.** E-002, E-005 and E-007 were all
   caused by trusting a published example over the installed library version.
   Three occurrences in one project is a pattern, not bad luck.
5. **Tools that fail as part of succeeding get misread.** E-009 cost a phase of
   work because an aborted commit and a successful one look similar at a glance.
6. **Environments carry dependencies you did not choose.** E-011 was caused by a
   package neither the project nor the developer installed, reached through a
   third library's dispatcher chain.
7. **Tests must force their preconditions, not inherit them.** E-013 passed and
   failed on different machines for the same code, because it asserted a
   property of the filesystem rather than of the software.
