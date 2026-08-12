# Metrics

Measured results only. Nothing here is estimated, projected, or copied from a
tutorial. Every number is reproducible with the command shown above it.

---

## Forecasting — price prediction

**Task.** Predict the price of a product at a given shop, from its own price
history. Series identity is `barcode@location_id`.

**Data.** Open Prices (Open Food Facts), ODbL. 287,749 rows in the export,
222,457 after filtering to EUR.

**Protocol.**
- Irregular observations (median gap 17 days) resampled to a **monthly grid**,
  median per bucket, forward-fill capped at 3 periods.
- **Strictly chronological, per-product** train/test split; verified by
  `Split.assert_no_leakage()` on every run.
- **Imputed rows excluded from scoring.** Forward-filled points are copies of
  their predecessor, so scoring on them rewards whichever model changes its
  prediction least.
- Metrics are the **median across series**, not the mean — prices span €1.50 to
  €30, and a mean would rank models by whichever handled one expensive product
  best.

---

### Run A — 38 series

```
python -m pulseiq.training.forecasting.train_forecast \
    --source open-prices --max-series 40 --no-prophet
```

Grid: 1,873 points · 38 series · 70.3% observed · median 44 points/series ·
2017-02 → 2026-08 · train 1,421 / test 452 · 266 evaluations in 7.1s

| model | MAE | RMSE | sMAPE | MASE | fallback |
|---|---|---|---|---|---|
| **naive_last** | **0.0484** | 0.0630 | 3.10% | 2.160 | 0% |
| moving_average_3 | 0.0518 | 0.0639 | 3.69% | 2.160 | 0% |
| moving_average_6 | 0.0528 | 0.0612 | 3.49% | 2.989 | 0% |
| arima_auto | 0.0640 | 0.0820 | 4.29% | 2.993 | 0% |
| drift | 0.0784 | 0.0931 | 5.32% | 4.583 | 0% |
| seasonal_naive_12 | 0.0824 | 0.1024 | 6.07% | 5.562 | 0% |
| mean | 0.2108 | 0.2204 | 10.75% | 10.974 | 0% |

---

### Run B — 96 series, including Prophet

```
python -m pulseiq.training.forecasting.train_forecast \
    --source open-prices --max-series 100
```

Grid: 3,756 points · 97 series · 69.7% observed · median 36 points/series ·
2010-07 → 2026-08 · train 2,799 / test 949 · 768 evaluations in 138.4s

| model | MAE | RMSE | sMAPE | MASE | fallback |
|---|---|---|---|---|---|
| **naive_last** | **0.0551** | 0.0649 | 3.35% | 2.068 | 0% |
| moving_average_6 | 0.0618 | 0.0779 | 3.77% | 2.468 | 0% |
| moving_average_3 | 0.0630 | 0.0804 | 3.69% | 2.073 | 0% |
| arima_auto | 0.0730 | 0.0862 | 3.87% | 2.629 | 0% |
| drift | 0.1115 | 0.1188 | 5.32% | 3.828 | 0% |
| seasonal_naive_12 | 0.1137 | 0.1266 | 6.24% | 4.390 | 0% |
| prophet | 0.1742 | 0.2128 | 7.46% | 6.337 | 0% |
| mean | 0.1831 | 0.1962 | 9.05% | 6.054 | 0% |

**Fallback rate is 0% for every model**, so no result here is a silently
degraded naive forecast wearing another model's name.

---

## Run C — horizon curve (the corrected evaluation)

```
python -m pulseiq.training.forecasting.train_forecast \
    --source open-prices --max-series 60 --min-observations 12 \
    --no-prophet --horizon-curve
```

Runs A and B used a single 20% holdout. On series spanning 2017–2026 that is a
~36-step-ahead forecast, which can only ever show that everything converges to
naive. This run replaces it with expanding-window origins at h = 1, 3, 6 and 12
months — the standard way forecast skill is reported.

Grid: 2,503 points · 56 series · 70.7% observed · median 42 points/series ·
2017-02 → 2026-08 · **3,430 evaluations**

### Median MAE by forecast horizon (months ahead)

| model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| **naive_last** | **0.0000** | **0.0100** | **0.0333** | **0.0418** |
| arima_auto | 0.0087 | 0.0205 | 0.0428 | 0.0623 |
| moving_average_3 | 0.0117 | 0.0300 | 0.0433 | 0.0633 |
| drift | 0.0124 | 0.0276 | 0.0628 | 0.1000 |
| moving_average_6 | 0.0283 | 0.0400 | 0.0527 | 0.0794 |
| seasonal_naive_12 | 0.1050 | 0.1142 | 0.1187 | 0.1914 |
| mean | 0.1557 | 0.1727 | 0.2073 | 0.3300 |

### Paired win rate against naive_last

Share of series-folds where the model had lower MAE than naive on the *same*
partition. 50% would mean indistinguishable.

| model | h=1 | h=3 | h=6 | h=12 |
|---|---|---|---|---|
| arima_auto | 19% | 22% | 22% | 29% |
| drift | 16% | 21% | 24% | 28% |
| moving_average_3 | 14% | 12% | 17% | 12% |
| moving_average_6 | 15% | 17% | 25% | 14% |
| seasonal_naive_12 | 12% | 11% | 17% | 9% |
| mean | 15% | 12% | 17% | 15% |

**No model exceeds 29% at any horizon.** Naive wins 71–91% of head-to-head
comparisons. This is consistent, not marginal.

---

## Headline finding

**The naive forecast is unbeaten at every horizon tested, and the reason is
visible in one number: median MAE at h=1 is exactly 0.0000.**

More than half the series do not change price from one month to the next. Where
the price is unchanged, "predict the last value" is not merely a good heuristic
— it is exactly correct, and there is no error left for a model to remove.

Three things support this being the correct result rather than a broken
evaluation:

1. **Prices are near-random-walk step functions.** For a random walk the naive
   forecast is provably optimal. A model beating it by a wide margin would be
   evidence of leakage, not skill.
2. **Error grows monotonically with horizon for every model**
   (naive: 0.0000 → 0.0100 → 0.0333 → 0.0418). Forecasting further ahead is
   harder, as it must be. An evaluation that did not show this would be suspect.
3. **The ranking is sensible.** Simple methods beat complex ones on short, noisy,
   level-shifting series. `seasonal_naive_12` and `mean` lose badly, which is
   what should happen to a yearly-seasonality assumption on 42-month series with
   no seasonal structure.

**Fallback rate is 0% for every model**, so no result here is a silently
degraded naive forecast reported under another model's name.

### What this means practically

The defensible claim from this project is **not** "I built a model that beats
the baseline." It is:

> Price forecasting on this data is a solved problem at short horizons — the
> naive forecast achieves zero median error one month ahead — so modelling
> effort belongs on *when* a price changes, not *what* it changes to.

That points directly at the discount-detection task below, which is where the
remaining signal is.

## Limitations of this evaluation

**Median MAE hides the tail.** Half the series have zero one-month error, so the
median reports 0.0000 while the series that *do* move are where all the error
lives. `reports/horizon_curve.csv` holds the full distribution; a follow-up
should report the conditional error given a price change occurred.

**Prophet is absent from Run C.** Four horizons x three folds x 56 series is
~670 Prophet fits, several minutes of compute for a model that placed last in
Run B. Excluded for iteration speed, not to flatter the result.

**Grocery, not electronics.** Open Prices covers supermarket products. Consumer
electronics discount more aggressively and more often, so the discount signal
would likely be stronger there. The licence trade-off (D-006) was accepted
knowingly.

---

## Discount classification — not yet run

The Open Prices ground-truth discount label (`price_is_discounted`) has a
**1.9% positive rate** in the export. That is too sparse for regression: a model
predicting zero everywhere would score an excellent MAE having learned nothing.

When built, this must be framed as **imbalanced binary classification** and
reported with precision, recall and PR-AUC, with the 1.9% base rate stated
alongside. Accuracy would be ~98% for a model that never predicts a discount.

---

---

## Sentiment — zero-shot vs LoRA fine-tuned

**Task.** Binary sentiment on Amazon product reviews (Electronics).

**Data.** Amazon Reviews'23 (McAuley Lab), non-commercial research use with
citation — see `docs/decision-log.md` D-007. 5,000 reviews, class-balanced
(the raw corpus is ~80% positive; without balancing an "always positive" model
scores 80% and appears to work). Labels derived from star ratings: 1-2 negative,
4-5 positive, 3-star dropped.

**Protocol.** 70/15/15 stratified split. Checkpoint selected on **validation**
macro-F1; the test set (n=750) scored exactly once, at the end, by both models.
Splits persisted to parquet so both models see byte-identical test data.

**Models.** Both are DistilBERT, so the measured delta isolates *domain
adaptation* rather than confounding it with model capacity.
- Zero-shot: `distilbert-base-uncased-finetuned-sst-2-english` (already
  sentiment-tuned, on movie reviews)
- Fine-tuned: `distilbert-base-uncased` + LoRA (r=16, alpha=32, `q_lin`/`v_lin`)

### Results (test set, n=750)

| metric | zero-shot | fine-tuned | delta |
|---|---|---|---|
| accuracy | 0.8880 | **0.9413** | +0.0533 |
| precision | 0.9505 | 0.9485 | −0.0020 |
| recall | 0.8187 | **0.9333** | **+0.1147** |
| f1 | 0.8797 | **0.9409** | +0.0612 |
| macro_f1 | 0.8875 | **0.9413** | +0.0539 |

**Error reduction: 47.6%** — nearly half the remaining errors eliminated.
At high accuracy this is the honest framing; "+5.3 points" understates it.

### Confusion matrix, fine-tuned

|  | pred neg | pred pos |
|---|---|---|
| **true neg** | 356 | 19 |
| **true pos** | 25 | 350 |

Majority-class floor on this balanced set: 50% accuracy.

### Training cost

| | |
|---|---|
| trainable parameters | **887,042 / 67,842,052 (1.31%)** |
| adapter size | **4.27 MB** (vs ~250 MB for a full checkpoint) |
| training time | **61 seconds**, 3 epochs, Colab T4 |
| cost | **£0** — free-tier GPU |

The adapter is small enough to commit, so the result is reproducible from the
repository. A full fine-tuned checkpoint would not be.

### What the numbers actually say

**The gain is almost entirely recall.** Precision moved −0.2 points while recall
moved +11.5. The zero-shot model was *missing negative reviews*: it was tuned on
SST-2, where negative sentiment is florid and explicit ("a disaster from start to
finish"). Product complaints are flat and factual ("battery died in a week"), and
the SST-2 model read too many of them as neutral-to-positive.

Fine-tuning taught it what dissatisfaction looks like in this domain, without
giving up precision. That is a specific, defensible domain-adaptation result
rather than a generic accuracy bump — and it is visible only because precision
and recall were reported separately.

### Limitations

**Labels are proxies.** Sentiment is derived from star ratings, and people leave
5 stars with complaints in the text. The achievable ceiling is set by label
noise, not model capacity. At 94.1% on labels that are perhaps ~95% faithful,
this run is close to saturated — further gains would likely be fitting the noise
rather than the signal, and should be treated with suspicion.

**5,000 reviews, one category.** Electronics only. A larger multi-category run
would test whether the adaptation generalises or is category-specific.

**Single run, no seed variance.** The reported numbers are from one seed. A
proper study would report mean and spread across several.
