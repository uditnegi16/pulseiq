"""Tests for sentiment dataset construction and classification metrics.

No torch, no network, no model downloads -- these run in CI. The fine-tuning
module itself is exercised on Colab; what is tested here is everything that
determines whether the fine-tuning result is *meaningful*: label derivation,
class balance, split integrity, and metric correctness.
"""

import numpy as np
import pandas as pd
import pytest

from pulseiq.evaluation.classification import (
    EmptyEvaluationError,
    compare,
    evaluate_classification,
    majority_baseline,
    to_metrics,
)
from pulseiq.training.sentiment.baseline_zeroshot import map_pipeline_label
from pulseiq.training.sentiment.dataset import (
    LABEL_NEGATIVE,
    LABEL_POSITIVE,
    clean_text,
    load_splits,
    prepare_rows,
    rating_to_label,
    save_splits,
    stratified_split,
)


def review_records(n=1000, positive_weight=0.8, seed=0):
    """Records shaped like the Amazon Reviews'23 schema."""
    rng = np.random.default_rng(seed)
    for i in range(n):
        rating = 5.0 if rng.random() < positive_weight else 1.0
        yield {
            "rating": rating,
            "title": f"Review title {i}",
            "text": "This product works really well and I am happy with it"
            if rating >= 4
            else "This product broke immediately and was a waste of money",
            "verified_purchase": True,
            "timestamp": 1600000000 + i,
        }


class TestRatingToLabel:
    @pytest.mark.parametrize(
        "rating,expected",
        [
            (1.0, LABEL_NEGATIVE),
            (2.0, LABEL_NEGATIVE),
            (4.0, LABEL_POSITIVE),
            (5.0, LABEL_POSITIVE),
            (1, LABEL_NEGATIVE),
            ("5", LABEL_POSITIVE),
        ],
    )
    def test_maps_stars_to_sentiment(self, rating, expected):
        assert rating_to_label(rating) == expected

    def test_three_stars_is_excluded(self):
        """3 stars is where the star-to-sentiment proxy is least trustworthy.
        Forcing it into a class teaches the model that mixed text has a definite
        label."""
        assert rating_to_label(3.0) is None

    @pytest.mark.parametrize("bad", [None, "", "not a number", float("nan")])
    def test_unparseable_ratings_are_excluded(self, bad):
        assert rating_to_label(bad) is None


class TestCleanText:
    def test_combines_title_and_body(self):
        assert clean_text("Great product", "Works well") == "Great product Works well"

    def test_title_alone_is_enough(self):
        """Titles carry real signal ('Waste of money') and are often the
        clearest part of a review."""
        assert clean_text("Waste of money", None) == "Waste of money"

    def test_collapses_whitespace(self):
        assert clean_text(None, "  too   many\n\nspaces  ") == "too many spaces"

    @pytest.mark.parametrize("title,text", [(None, None), ("", ""), ("   ", "\n")])
    def test_empty_inputs_return_none(self, title, text):
        assert clean_text(title, text) is None

    def test_truncates_very_long_text(self):
        assert len(clean_text(None, "word " * 2000)) <= 2000


class TestPrepareRows:
    def test_balances_classes(self):
        """Amazon reviews are ~80% positive. Without balancing, a model that
        always predicts positive scores 80% and appears to work."""
        rows, _ = prepare_rows(review_records(3000, positive_weight=0.8), max_rows=400)
        frame = pd.DataFrame(rows)
        assert frame["label"].mean() == pytest.approx(0.5, abs=0.01)

    def test_unbalanced_mode_preserves_skew(self):
        rows, _ = prepare_rows(
            review_records(2000, positive_weight=0.8), max_rows=400, balance=False
        )
        assert pd.DataFrame(rows)["label"].mean() > 0.6

    def test_drops_neutral_ratings(self):
        records = [{"rating": 3.0, "title": "Okay", "text": "It is fine I guess maybe"}] * 50
        rows, stats = prepare_rows(iter(records), max_rows=100)
        assert rows == []
        assert stats.dropped["neutral_or_unparseable_rating"] == 50

    def test_drops_duplicate_text(self):
        record = {"rating": 5.0, "title": "Good", "text": "This is a great product overall"}
        rows, stats = prepare_rows(iter([record] * 10), max_rows=100)
        assert len(rows) == 1
        assert stats.dropped["duplicate_text"] == 9

    def test_drops_very_short_reviews(self):
        records = [{"rating": 5.0, "title": None, "text": "Good"}] * 20
        rows, stats = prepare_rows(iter(records), max_rows=50, min_words=5)
        assert rows == []
        assert stats.dropped["too_short"] == 20

    def test_respects_max_rows(self):
        rows, _ = prepare_rows(review_records(5000), max_rows=200)
        assert len(rows) <= 200

    def test_stats_report_keep_rate(self):
        _, stats = prepare_rows(review_records(1000), max_rows=100)
        assert "keep_rate" in stats.summary()

    def test_empty_input(self):
        rows, stats = prepare_rows(iter([]), max_rows=100)
        assert rows == []
        assert stats.total_seen == 0


class TestStratifiedSplit:
    def test_preserves_class_balance(self):
        rows, _ = prepare_rows(review_records(3000), max_rows=1000)
        frame = pd.DataFrame(rows)
        train, val, test = stratified_split(frame)
        for part in (train, val, test):
            assert part["label"].mean() == pytest.approx(0.5, abs=0.05)

    def test_three_way_split_sizes(self):
        rows, _ = prepare_rows(review_records(3000), max_rows=1000)
        train, val, test = stratified_split(pd.DataFrame(rows), test_size=0.15, val_size=0.15)
        assert len(train) + len(val) + len(test) == 1000
        assert len(test) == pytest.approx(150, abs=5)
        assert len(val) == pytest.approx(150, abs=5)

    def test_splits_are_disjoint(self):
        """Overlap between train and test would invalidate every reported number."""
        rows, _ = prepare_rows(review_records(3000), max_rows=600)
        train, val, test = stratified_split(pd.DataFrame(rows))
        train_texts = set(train["text"])
        assert not train_texts & set(test["text"])
        assert not train_texts & set(val["text"])

    def test_deterministic_given_a_seed(self):
        rows, _ = prepare_rows(review_records(2000), max_rows=400)
        frame = pd.DataFrame(rows)
        first = stratified_split(frame, seed=1)[2]["text"].tolist()
        second = stratified_split(frame, seed=1)[2]["text"].tolist()
        assert first == second

    def test_empty_frame(self):
        empty = pd.DataFrame(columns=["text", "label"])
        train, val, test = stratified_split(empty)
        assert all(part.empty for part in (train, val, test))


class TestSaveLoadSplits:
    def test_round_trip(self, tmp_path):
        rows, _ = prepare_rows(review_records(1000), max_rows=200)
        train, val, test = stratified_split(pd.DataFrame(rows))
        save_splits(train, val, test, tmp_path)
        loaded_train, loaded_val, loaded_test = load_splits(tmp_path)
        assert len(loaded_train) == len(train)
        assert loaded_test["text"].tolist() == test["text"].tolist()

    def test_missing_splits_raise_with_instructions(self, tmp_path):
        """Persisted splits are what keep the baseline and fine-tune scored on
        byte-identical test data."""
        with pytest.raises(FileNotFoundError, match="Run dataset preparation"):
            load_splits(tmp_path)


class TestMapPipelineLabel:
    @pytest.mark.parametrize(
        "raw,expected",
        [("POSITIVE", 1), ("NEGATIVE", 0), ("positive", 1), ("LABEL_1", 1), ("LABEL_0", 0)],
    )
    def test_maps_known_labels(self, raw, expected):
        assert map_pipeline_label(raw) == expected

    @pytest.mark.parametrize("raw", [None, "UNKNOWN", ""])
    def test_unknown_labels_return_none(self, raw):
        assert map_pipeline_label(raw) is None


class TestClassificationMetrics:
    def test_perfect_predictions(self):
        y = [0, 1, 0, 1]
        metrics = evaluate_classification(y, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["f1"] == 1.0

    def test_confusion_matrix_cells(self):
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        m = to_metrics(evaluate_classification(y_true, y_pred))
        assert (m.true_negatives, m.false_positives, m.false_negatives, m.true_positives) == (
            1,
            1,
            1,
            1,
        )
        assert m.confusion_matrix == [[1, 1], [1, 1]]

    def test_all_one_class_predicted_does_not_crash(self):
        """precision is undefined when nothing is predicted positive; 0 is the
        honest answer."""
        metrics = evaluate_classification([0, 1, 0, 1], [0, 0, 0, 0])
        assert metrics["precision"] == 0.0
        assert metrics["accuracy"] == 0.5

    def test_empty_raises(self):
        with pytest.raises(EmptyEvaluationError):
            evaluate_classification([], [])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            evaluate_classification([0, 1], [0])

    def test_negative_recall_is_reported(self):
        """First thing to collapse when a model drifts to the majority class,
        and overall accuracy hides it completely."""
        m = to_metrics(evaluate_classification([0, 0, 0, 1], [1, 1, 0, 1]))
        assert m.negative_recall == pytest.approx(1 / 3)

    def test_confusion_table_is_readable(self):
        table = to_metrics(evaluate_classification([0, 1], [0, 1])).confusion_table()
        assert "pred neg" in table and "true pos" in table


class TestMajorityBaseline:
    def test_imbalanced_data_gives_a_high_accuracy_floor(self):
        """The number that stops '98% accuracy' being reported as a result."""
        y = [1] * 98 + [0] * 2
        metrics = majority_baseline(y)
        assert metrics["accuracy"] == 0.98
        assert metrics["macro_f1"] < 0.6  # exposes what accuracy hides

    def test_balanced_data_gives_a_50_percent_floor(self):
        assert majority_baseline([0] * 50 + [1] * 50)["accuracy"] == 0.5

    def test_empty_raises(self):
        with pytest.raises(EmptyEvaluationError):
            majority_baseline([])


class TestCompare:
    def test_reports_deltas(self):
        before = {"accuracy": 0.80, "f1": 0.79, "macro_f1": 0.79, "precision": 0.8, "recall": 0.8}
        after = {"accuracy": 0.90, "f1": 0.90, "macro_f1": 0.90, "precision": 0.9, "recall": 0.9}
        result = compare(before, after)
        assert result["accuracy_delta"] == pytest.approx(0.10)

    def test_error_reduction_is_the_honest_framing_at_high_accuracy(self):
        """91% -> 94% is '+3 points', which understates removing a third of the
        remaining errors."""
        before = {"accuracy": 0.91}
        after = {"accuracy": 0.94}
        assert compare(before, after)["error_reduction_pct"] == pytest.approx(33.33, abs=0.1)

    def test_no_division_by_zero_at_perfect_baseline(self):
        assert compare({"accuracy": 1.0}, {"accuracy": 1.0})["error_reduction_pct"] == 0.0
