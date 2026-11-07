import pytest
from ml.sweep_thresholds import (
    evaluate_grid_point,
    bootstrap_ci,
    paired_bootstrap_diff,
    parse_grid,
    build_parser,
    DEFAULT_MIN_RET_RANGE,
    DEFAULT_STANCE_RANGE,
)


def _item(sup=0.0, ref=0.0, score=0.5):
    return {
        "retrieval_score": score,
        "probs": {
            "SUPPORTED": sup,
            "REFUTED": ref,
            "NOT_ENOUGH_INFO": round(1.0 - sup - ref, 6),
        },
    }


def test_evaluate_grid_point_hand_made_fixture():
    # 3 hand-crafted claims:
    # 1. Strong supported item (score=0.20, sup=0.80) -> Target: SUPPORTED
    # 2. Strong refuted item (score=0.20, ref=0.80) -> Target: REFUTED
    # 3. Weak item (score=0.03, sup=0.30) -> Target: INSUFFICIENT
    cached_items = [
        [_item(sup=0.80, score=0.20)],
        [_item(ref=0.80, score=0.20)],
        [_item(sup=0.30, score=0.03)],
    ]
    targets = ["SUPPORTED", "REFUTED", "INSUFFICIENT"]

    # At MR=0.05, ST=0.60:
    # Claim 1: SUPPORTED (clears MR 0.05, ST 0.60)
    # Claim 2: REFUTED (clears MR 0.05, ST 0.60)
    # Claim 3: INSUFFICIENT (retrieval 0.03 < 0.05)
    # All 3 correct -> accuracy=1.0, macro_f1=1.0
    res = evaluate_grid_point(cached_items, targets, mr=0.05, st=0.60)
    assert res["accuracy"] == pytest.approx(1.0)
    assert res["macro_f1"] == pytest.approx(1.0)
    assert res["recall_SUPPORTED"] == pytest.approx(1.0)
    assert res["recall_REFUTED"] == pytest.approx(1.0)
    assert res["recall_INSUFFICIENT"] == pytest.approx(1.0)
    assert res["share_SUPPORTED"] == pytest.approx(1 / 3)
    assert res["share_REFUTED"] == pytest.approx(1 / 3)
    assert res["share_INSUFFICIENT"] == pytest.approx(1 / 3)
    assert res["share_CONFLICTING"] == pytest.approx(0.0)

    # If ST is raised to 0.90, claims 1 and 2 become INSUFFICIENT
    res_high_st = evaluate_grid_point(cached_items, targets, mr=0.05, st=0.90)
    assert res_high_st["share_INSUFFICIENT"] == pytest.approx(1.0)
    assert res_high_st["accuracy"] == pytest.approx(1 / 3)


def test_bootstrap_ci():
    targets = ["SUPPORTED", "REFUTED", "INSUFFICIENT"] * 10
    preds = ["SUPPORTED", "REFUTED", "INSUFFICIENT"] * 10
    acc_ci, f1_ci = bootstrap_ci(targets, preds, n_resamples=100, seed=42)

    assert 0.0 <= acc_ci[0] <= acc_ci[1] <= 1.0
    assert 0.0 <= f1_ci[0] <= f1_ci[1] <= 1.0
    # Perfect match should yield near-1.0 interval
    assert acc_ci[0] == pytest.approx(1.0)
    assert f1_ci[0] == pytest.approx(1.0)


def test_parse_grid_and_parser():
    # Test parse_grid default and custom values
    assert parse_grid(None, DEFAULT_MIN_RET_RANGE) == DEFAULT_MIN_RET_RANGE
    assert parse_grid("", DEFAULT_MIN_RET_RANGE) == DEFAULT_MIN_RET_RANGE
    assert parse_grid("0.05, 0.1, 0.2", DEFAULT_MIN_RET_RANGE) == [0.05, 0.1, 0.2]
    assert parse_grid([0.1, 0.25], DEFAULT_MIN_RET_RANGE) == [0.1, 0.25]

    # Test build_parser with opt-in grid flags
    parser = build_parser()
    args = parser.parse_args([
        "--checkpoint", "dummy.pt",
        "--out", "out.csv",
        "--label", "test",
        "--mr-grid", "0.01,0.05,0.1",
        "--st-grid", "0.4,0.6,0.8",
        "--top-k", "5",
        "--corpus", "data/custom_corpus.jsonl"
    ])
    assert parse_grid(args.mr_grid, DEFAULT_MIN_RET_RANGE) == [0.01, 0.05, 0.1]
    assert parse_grid(args.st_grid, DEFAULT_STANCE_RANGE) == [0.4, 0.6, 0.8]
    assert args.top_k == 5
    assert args.corpus == "data/custom_corpus.jsonl"

    # Test parser with paired comparison args
    paired_args = parser.parse_args([
        "--checkpoint", "dummy.pt",
        "--paired",
        "--mr-a", "0.05",
        "--st-a", "0.60",
        "--mr-b", "0.30",
        "--st-b", "0.55",
        "--paired-out", "results/paired.txt",
    ])
    assert paired_args.paired is True
    assert paired_args.mr_a == 0.05
    assert paired_args.st_b == 0.55
    assert paired_args.paired_out == "results/paired.txt"



def test_run_sweep_refuses_test_set():
    from ml.sweep_thresholds import run_sweep
    with pytest.raises(ValueError, match="strictly prohibited on test sets"):
        run_sweep("dummy.pt", "data/claim_level/test_claims.jsonl", "Test-Run", "dummy.csv")


def test_paired_bootstrap_diff():
    targets = ["SUPPORTED", "REFUTED", "INSUFFICIENT", "SUPPORTED"]
    # A gets 1 right (index 0), 3 wrong
    preds_a = ["SUPPORTED", "SUPPORTED", "SUPPORTED", "REFUTED"]
    # B gets 3 right (index 0, 1, 2), 1 wrong
    preds_b = ["SUPPORTED", "REFUTED", "INSUFFICIENT", "REFUTED"]

    res = paired_bootstrap_diff(targets, preds_a, preds_b, n_resamples=200, seed=42)
    assert res["n"] == 4
    assert res["acc_a"] == 0.25
    assert res["acc_b"] == 0.75
    assert res["diff"] == pytest.approx(0.50)
    assert res["b_wins"] == 2  # claims 1 and 2
    assert res["a_wins"] == 0
    assert res["both_correct"] == 1  # claim 0
    assert res["both_incorrect"] == 1  # claim 3
    assert res["ci_95"][0] <= res["diff"] <= res["ci_95"][1]



