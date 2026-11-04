"""Tests for ml/baselines.py.

Checks:
- Both baselines run on a tiny fixture dataset.
- The produced metrics.json has the exact same key set as a neural run.
- Accuracy is in (0, 1].
"""
import json
import tempfile
from pathlib import Path

import pytest

from ml.baselines import run_majority, run_tfidf_logreg, _write_report, _metrics_dict
from ml.data import LABELS

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

# Tiny synthetic data covering all 3 VERITAS labels.
# 18 rows: 6 per class, tiny texts so the test is fast.
_TRAIN_ROWS = [
    # SUPPORTED (label index 0)
    {"claim": "Paris is the capital of France.", "evidence": "The capital of France is Paris.", "label": "SUPPORTED"},
    {"claim": "Water boils at 100 degrees.", "evidence": "Water boils at 100 degrees Celsius.", "label": "SUPPORTED"},
    {"claim": "The earth orbits the sun.", "evidence": "Earth revolves around the sun.", "label": "SUPPORTED"},
    {"claim": "Spain won Euro 2024.", "evidence": "Spain defeated England in the Euro 2024 final.", "label": "SUPPORTED"},
    {"claim": "Bitcoin surpassed 70000 dollars.", "evidence": "Bitcoin price exceeded 70000 USD in 2024.", "label": "SUPPORTED"},
    {"claim": "India has a large population.", "evidence": "India is the most populous country in the world.", "label": "SUPPORTED"},
    # REFUTED (label index 1)
    {"claim": "Paris is the capital of Germany.", "evidence": "Berlin is the capital of Germany.", "label": "REFUTED"},
    {"claim": "Water boils at 50 degrees.", "evidence": "Water boils at 100 degrees Celsius.", "label": "REFUTED"},
    {"claim": "The moon orbits Mars.", "evidence": "The moon orbits the Earth.", "label": "REFUTED"},
    {"claim": "England won Euro 2024.", "evidence": "Spain defeated England in the Euro 2024 final.", "label": "REFUTED"},
    {"claim": "Bitcoin fell below 20000.", "evidence": "Bitcoin price exceeded 70000 USD in 2024.", "label": "REFUTED"},
    {"claim": "China has a small population.", "evidence": "China has over 1.4 billion people.", "label": "REFUTED"},
    # NOT_ENOUGH_INFO (label index 2)
    {"claim": "Paris is famous for art.", "evidence": "Germany has an advanced car industry.", "label": "NOT_ENOUGH_INFO"},
    {"claim": "Water is essential for life.", "evidence": "The moon has no atmosphere.", "label": "NOT_ENOUGH_INFO"},
    {"claim": "The sun is a star.", "evidence": "Football is played globally.", "label": "NOT_ENOUGH_INFO"},
    {"claim": "Elections are important.", "evidence": "The Amazon rainforest covers millions of hectares.", "label": "NOT_ENOUGH_INFO"},
    {"claim": "Technology is advancing rapidly.", "evidence": "Inflation affects purchasing power.", "label": "NOT_ENOUGH_INFO"},
    {"claim": "Climate change is a global concern.", "evidence": "The stock market fluctuates daily.", "label": "NOT_ENOUGH_INFO"},
]

# Use the same 18 rows for dev and test in the fixture (fine for unit tests;
# accuracy just needs to be in (0, 1]).
_DEV_ROWS = _TRAIN_ROWS[:]
_TEST_ROWS = _TRAIN_ROWS[:]


def _write_jsonl(path: Path, rows: list) -> None:
    import json
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def _extract(rows):
    from ml.data import L2I
    claims   = [r["claim"]    for r in rows]
    evidences = [r["evidence"] for r in rows]
    labels   = [L2I[r["label"]] for r in rows]
    return claims, evidences, labels


@pytest.fixture()
def tiny_data():
    tr_c, tr_e, tr_y = _extract(_TRAIN_ROWS)
    dv_c, dv_e, dv_y = _extract(_DEV_ROWS)
    ts_c, ts_e, ts_y = _extract(_TEST_ROWS)
    return (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y)


# ---------------------------------------------------------------------------
# Reference key set from an existing neural metrics.json
# ---------------------------------------------------------------------------

_NEURAL_KEYS = {
    "name", "arch", "seed", "n_params", "optimizer", "activation",
    "dropout", "weight_decay", "epochs_run", "train_seconds",
    "best_dev_macro_f1", "loss", "accuracy", "macro_f1", "precision",
    "recall", "confusion_matrix", "latency_ms", "labels",
}


def _keys_from_neural_report() -> set:
    """Read the actual on-disk neural metrics.json if it exists."""
    p = Path("reports/bilstm_attn_s42/metrics.json")
    if p.exists():
        return set(json.loads(p.read_text()).keys())
    return _NEURAL_KEYS  # fall back to the hard-coded set


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMajorityBaseline:
    def test_runs_and_returns_dict(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_majority(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="majority_s0",
        )
        assert isinstance(result, dict)

    def test_accuracy_in_range(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_majority(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="majority_s0",
        )
        assert 0 < result["accuracy"] <= 1.0

    def test_metrics_json_same_keys_as_neural_run(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_majority(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="majority_s0",
        )
        expected_keys = _keys_from_neural_report()
        assert set(result.keys()) == expected_keys, (
            f"Key mismatch.\n  Missing: {expected_keys - set(result.keys())}\n"
            f"  Extra:   {set(result.keys()) - expected_keys}"
        )

    def test_write_report_creates_file(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_majority(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="majority_s0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "majority_s0"
            _write_report(out, result)
            assert (out / "metrics.json").exists()
            loaded = json.loads((out / "metrics.json").read_text())
            assert set(loaded.keys()) == _keys_from_neural_report()

    def test_confusion_matrix_shape(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_majority(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="majority_s0",
        )
        cm = result["confusion_matrix"]
        assert len(cm) == 3 and all(len(row) == 3 for row in cm)

    def test_labels_field(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_majority(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="majority_s0",
        )
        assert result["labels"] == LABELS


class TestTfidfLogregBaseline:
    def test_runs_and_returns_dict(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_tfidf_logreg(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="tfidf_logreg_s0",
        )
        assert isinstance(result, dict)

    def test_accuracy_in_range(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_tfidf_logreg(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="tfidf_logreg_s0",
        )
        assert 0 < result["accuracy"] <= 1.0

    def test_metrics_json_same_keys_as_neural_run(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_tfidf_logreg(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="tfidf_logreg_s0",
        )
        expected_keys = _keys_from_neural_report()
        assert set(result.keys()) == expected_keys, (
            f"Key mismatch.\n  Missing: {expected_keys - set(result.keys())}\n"
            f"  Extra:   {set(result.keys()) - expected_keys}"
        )

    def test_write_report_creates_file(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_tfidf_logreg(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="tfidf_logreg_s0",
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "tfidf_logreg_s0"
            _write_report(out, result)
            assert (out / "metrics.json").exists()
            loaded = json.loads((out / "metrics.json").read_text())
            assert set(loaded.keys()) == _keys_from_neural_report()

    def test_confusion_matrix_shape(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_tfidf_logreg(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="tfidf_logreg_s0",
        )
        cm = result["confusion_matrix"]
        assert len(cm) == 3 and all(len(row) == 3 for row in cm)

    def test_labels_field(self, tiny_data):
        (tr_c, tr_e, tr_y), (dv_c, dv_e, dv_y), (ts_c, ts_e, ts_y) = tiny_data
        result = run_tfidf_logreg(
            tr_c, tr_e, tr_y,
            dv_c, dv_e, dv_y,
            ts_c, ts_e, ts_y,
            name="tfidf_logreg_s0",
        )
        assert result["labels"] == LABELS
