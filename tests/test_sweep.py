import pytest
from ml.sweep_thresholds import evaluate_grid_point, bootstrap_ci


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
