from backend.app.config import MIN_RETRIEVAL, STANCE_THRESHOLD
from backend.app.services.verifier import decide


def _make_item(sup=0.0, ref=0.0, score=0.5):
    return {
        "retrieval_score": score,
        "probs": {
            "SUPPORTED": sup,
            "REFUTED": ref,
            "NOT_ENOUGH_INFO": round(1.0 - sup - ref, 6),
        },
    }


def test_decide_default_behavior():
    # Fixed hand-made examples matching module defaults
    item_sup = [_make_item(sup=0.65, score=0.10)]
    verdict, conf = decide(item_sup)
    assert verdict == "SUPPORTED"
    assert conf == 0.65

    # Retrieval score 0.04 (< default MIN_RETRIEVAL 0.05) -> INSUFFICIENT
    item_low_ret = [_make_item(sup=0.90, score=0.04)]
    verdict, _ = decide(item_low_ret)
    assert verdict == "INSUFFICIENT"

    # Stance prob 0.55 (< default STANCE_THRESHOLD 0.60) -> INSUFFICIENT
    item_low_stance = [_make_item(sup=0.55, score=0.10)]
    verdict, _ = decide(item_low_stance)
    assert verdict == "INSUFFICIENT"


def test_decide_overriding_changes_outcome():
    # Lowering min_retrieval allows previously rejected item to pass
    item_low_ret = [_make_item(sup=0.90, score=0.04)]
    assert decide(item_low_ret)[0] == "INSUFFICIENT"
    verdict, conf = decide(item_low_ret, min_retrieval=0.03)
    assert verdict == "SUPPORTED"
    assert conf == 0.90

    # Raising min_retrieval filters out previously passing item
    item_normal = [_make_item(sup=0.80, score=0.08)]
    assert decide(item_normal)[0] == "SUPPORTED"
    assert decide(item_normal, min_retrieval=0.10)[0] == "INSUFFICIENT"

    # Lowering stance_threshold accepts weaker stance
    item_weak_sup = [_make_item(sup=0.55, score=0.10)]
    assert decide(item_weak_sup)[0] == "INSUFFICIENT"
    verdict, conf = decide(item_weak_sup, stance_threshold=0.50)
    assert verdict == "SUPPORTED"
    assert conf == 0.55

    # Raising stance_threshold rejects moderate stance
    item_mod_sup = [_make_item(sup=0.65, score=0.10)]
    assert decide(item_mod_sup)[0] == "SUPPORTED"
    assert decide(item_mod_sup, stance_threshold=0.70)[0] == "INSUFFICIENT"
