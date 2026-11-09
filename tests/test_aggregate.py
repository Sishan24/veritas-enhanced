"""Tests for ml/aggregate.py."""
import pytest

from ml.aggregate import (
    aggregate_runs,
    compute_mean_std,
    format_mean_std,
    parse_run_dir_name,
)


def test_compute_mean_std_known_values():
    # Known dataset 1: [10.0, 20.0, 30.0]
    # mean = 20.0, sample variance (ddof=1) = (100 + 0 + 100) / 2 = 100.0, sample std = 10.0
    mean, std = compute_mean_std([10.0, 20.0, 30.0])
    assert mean == pytest.approx(20.0)
    assert std == pytest.approx(10.0)

    # Known dataset 2: [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
    # sum = 40.0, mean = 5.0
    # sq_diffs = [9, 1, 1, 1, 0, 0, 4, 16] sum = 32
    # sample std = sqrt(32 / 7) = 2.1380899...
    mean2, std2 = compute_mean_std([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    assert mean2 == pytest.approx(5.0)
    assert std2 == pytest.approx((32.0 / 7.0) ** 0.5)


def test_compute_mean_std_n1_case():
    # n = 1: mean is the value, sample std (ddof=1) is undefined / None
    mean, std = compute_mean_std([0.85])
    assert mean == 0.85
    assert std is None

    # format_mean_std must show "n/a" for std, NEVER 0
    formatted = format_mean_std(mean, std)
    assert formatted == "0.8500 ± n/a"
    assert "0.0000" not in formatted


def test_compute_mean_std_empty():
    mean, std = compute_mean_std([])
    assert mean is None
    assert std is None
    assert format_mean_std(mean, std) == "n/a"


def test_deterministic_baseline_handling():
    # Create hand-made runs where seed 0 and seed 1 are identical
    synthetic_runs = []
    # Majority baseline: seed 0 and seed 1 have identical values
    baseline_metrics = {
        "pair": 0.34,
        "passage_ss0": 0.35,
        "passage_ss1": 0.36,
        "passage_ss2": 0.37,
    }
    for split_tag, acc in baseline_metrics.items():
        for ms in (0, 1):
            split = "pair" if split_tag == "pair" else "passage"
            ss = None if split_tag == "pair" else int(split_tag.replace("passage_ss", ""))
            synthetic_runs.append({
                "dir_name": f"majority_h0_{split_tag}_ms{ms}",
                "arch": "majority",
                "hidden": 0,
                "split": split,
                "split_seed": ss,
                "split_tag": split_tag,
                "model_seed": ms,
                "n_params": 0,
                "accuracy": acc,
                "macro_f1": acc / 2.0,
            })

    # Add a neural model with ms0 and ms1 differing
    for split_tag in ["pair", "passage_ss0", "passage_ss1", "passage_ss2"]:
        split = "pair" if split_tag == "pair" else "passage"
        ss = None if split_tag == "pair" else int(split_tag.replace("passage_ss", ""))
        synthetic_runs.append({
            "dir_name": f"bilstm_h128_{split_tag}_ms0",
            "arch": "bilstm",
            "hidden": 128,
            "split": split,
            "split_seed": ss,
            "split_tag": split_tag,
            "model_seed": 0,
            "n_params": 500000,
            "accuracy": 0.90,
            "macro_f1": 0.90,
        })
        synthetic_runs.append({
            "dir_name": f"bilstm_h128_{split_tag}_ms1",
            "arch": "bilstm",
            "hidden": 128,
            "split": split,
            "split_seed": ss,
            "split_tag": split_tag,
            "model_seed": 1,
            "n_params": 500000,
            "accuracy": 0.92,
            "macro_f1": 0.92,
        })

    # Add lstm_h256 for capacity comparison
    for split_tag in ["pair", "passage_ss0", "passage_ss1", "passage_ss2"]:
        split = "pair" if split_tag == "pair" else "passage"
        ss = None if split_tag == "pair" else int(split_tag.replace("passage_ss", ""))
        synthetic_runs.append({
            "dir_name": f"lstm_h256_{split_tag}_ms0",
            "arch": "lstm",
            "hidden": 256,
            "split": split,
            "split_seed": ss,
            "split_tag": split_tag,
            "model_seed": 0,
            "n_params": 750000,
            "accuracy": 0.95,
            "macro_f1": 0.95,
        })
        synthetic_runs.append({
            "dir_name": f"lstm_h256_{split_tag}_ms1",
            "arch": "lstm",
            "hidden": 256,
            "split": split,
            "split_seed": ss,
            "split_tag": split_tag,
            "model_seed": 1,
            "n_params": 750000,
            "accuracy": 0.97,
            "macro_f1": 0.97,
        })

    agg = aggregate_runs(synthetic_runs)

    # 1. Deterministic baseline: pair split has n=1 (not 2)
    maj_pair = next(r for r in agg["pair_rows"] if r["arch"] == "majority")
    assert maj_pair["n"] == 1
    assert maj_pair["raw_acc_mean"] == pytest.approx(0.34)
    assert maj_pair["raw_acc_std"] is None
    assert "± n/a" in maj_pair["accuracy mean ± std"]

    # 2. Deterministic baseline: passage pooled has n=3 (spread across the 3 split seeds), NOT n=6
    maj_pass = next(r for r in agg["passage_pooled_rows"] if r["arch"] == "majority")
    assert maj_pass["n"] == 3
    # Mean of 0.35, 0.36, 0.37 is 0.36
    assert maj_pass["raw_acc_mean"] == pytest.approx(0.36)
    # Sample std of [0.35, 0.36, 0.37]: mean=0.36, diffs=[-0.01, 0, 0.01], var=(0.0001+0.0001)/2 = 0.0001, std = 0.01
    assert maj_pass["raw_acc_std"] == pytest.approx(0.01)

    # 3. Neural model: pair split has n=2
    bilstm_pair = next(r for r in agg["pair_rows"] if r["arch"] == "bilstm")
    assert bilstm_pair["n"] == 2
    assert bilstm_pair["raw_acc_mean"] == pytest.approx(0.91)
    # std of [0.90, 0.92] with ddof=1: sqrt((0.01^2 + 0.01^2)/1) = sqrt(0.0002) = 0.014142...
    assert bilstm_pair["raw_acc_std"] == pytest.approx((0.0002) ** 0.5)

    # 4. Neural model: passage pooled has n=6
    bilstm_pass = next(r for r in agg["passage_pooled_rows"] if r["arch"] == "bilstm")
    assert bilstm_pass["n"] == 6


def test_parse_run_dir_name():
    p1 = parse_run_dir_name("bilstm_attn_h128_passage_ss1_ms0")
    assert p1["arch"] == "bilstm_attn"
    assert p1["hidden"] == 128
    assert p1["split"] == "passage"
    assert p1["split_seed"] == 1
    assert p1["model_seed"] == 0

    p2 = parse_run_dir_name("tfidf_logreg_h0_pair_ms1")
    assert p2["arch"] == "tfidf_logreg"
    assert p2["hidden"] == 0
    assert p2["split"] == "pair"
    assert p2["split_seed"] is None
    assert p2["model_seed"] == 1
