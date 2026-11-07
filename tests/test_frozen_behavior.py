"""Regression test verifying that existing model architectures, decide(), and predict() remain frozen."""
import json
from pathlib import Path
import pytest
import torch

from backend.app.services.verifier import decide, VerifierService
from ml.models import Verifier

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_frozen_architectures():
    fixture_path = FIXTURES_DIR / "frozen_model_outputs.json"
    assert fixture_path.exists(), f"Missing fixture: {fixture_path}"

    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    claim = torch.tensor(data["claim"], dtype=torch.long)
    claim_mask = claim != 0
    ev = torch.tensor(data["ev"], dtype=torch.long)
    ev_mask = ev != 0

    for arch, expected in data["archs"].items():
        torch.manual_seed(42)
        model = Verifier(arch=arch, vocab_size=50, emb_dim=32, hidden=64, dropout=0.0)
        model.eval()

        with torch.no_grad():
            if arch == "bilstm_attn":
                logits, attn = model(claim, claim_mask, ev, ev_mask, return_attention=True)
                expected_attn = expected["attn"]
                for i in range(len(expected_attn)):
                    for j in range(len(expected_attn[i])):
                        assert attn[i, j].item() == pytest.approx(expected_attn[i][j], abs=1e-4)
            else:
                logits = model(claim, claim_mask, ev, ev_mask)

        expected_logits = expected["logits"]
        for i in range(len(expected_logits)):
            for j in range(len(expected_logits[i])):
                assert logits[i, j].item() == pytest.approx(expected_logits[i][j], abs=1e-4)


def test_frozen_decide():
    fixture_path = FIXTURES_DIR / "frozen_decide_outputs.json"
    assert fixture_path.exists(), f"Missing fixture: {fixture_path}"

    test_cases = json.loads(fixture_path.read_text(encoding="utf-8"))
    for tc in test_cases:
        verdict, conf = decide(tc["items"])
        assert verdict == tc["expected_verdict"], f"Failed for case {tc['name']}"
        assert conf == pytest.approx(tc["expected_confidence"], abs=1e-5), f"Confidence mismatch for {tc['name']}"


def test_frozen_predict():
    fixture_path = FIXTURES_DIR / "frozen_predict_outputs.json"
    if not fixture_path.exists():
        pytest.skip(f"Fixture {fixture_path} not found.")

    data = json.loads(fixture_path.read_text(encoding="utf-8"))
    ckpt_path = Path(data["checkpoint"])
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint {ckpt_path} not found locally (fresh clone / CI mode).")

    vs = VerifierService(str(ckpt_path))
    if not vs.ready:
        pytest.skip("VerifierService failed to load checkpoint.")

    out = vs.predict(data["claim"], data["passages"])
    expected = data["expected"]
    assert len(out) == len(expected)

    for actual_item, exp_item in zip(out, expected):
        assert actual_item["stance"] == exp_item["stance"]
        for k in ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]:
            assert actual_item["probs"][k] == pytest.approx(exp_item["probs"][k], abs=1e-4)
        if exp_item["attention"] is not None:
            assert len(actual_item["attention"]) == len(exp_item["attention"])
            for a_act, a_exp in zip(actual_item["attention"], exp_item["attention"]):
                assert a_act["token"] == a_exp["token"]
                assert a_act["weight"] == pytest.approx(a_exp["weight"], abs=1e-4)
