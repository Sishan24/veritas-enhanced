"""Tests for enhancement branch improvements:
- Token highlight synchronization on edge cases
- Explainability occlusion check
- Data constant synchronization
- Single-encoding claim optimization
"""
from pathlib import Path
import pytest
import torch

from backend.app.services.verifier import (
    MAX_CLAIM,
    MAX_EV,
    VerifierService,
)
from ml.data import DEFAULT_MAX_CLAIM, DEFAULT_MAX_EV, Vocab, tokenize


def test_shared_constants():
    """Verifier and dataset constants must reference the exact same definition."""
    assert MAX_CLAIM == DEFAULT_MAX_CLAIM == 32
    assert MAX_EV == DEFAULT_MAX_EV == 64


def test_tokenize_edge_cases():
    """Punctuation-only text returns empty from tokenize, handled safely."""
    assert tokenize("??? !!! ...") == []
    assert tokenize("Normal claim 123") == ["normal", "claim", "123"]


def test_predict_empty_text_token_sync():
    """Predict must not fail or desync tokens when evidence has only punctuation."""
    ckpt_path = Path("ml/checkpoints/bilstm_attn.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint bilstm_attn.pt not found")

    vs = VerifierService(str(ckpt_path))
    assert vs.ready

    # Normal text
    out = vs.predict("Paris is in France.", ["Paris is the capital of France."])
    assert len(out) == 1
    assert out[0]["attention"] is not None
    assert len(out[0]["attention"]) > 0

    # Punctuation-only evidence: tokenize gives [], encode gives [UNK]
    out_punct = vs.predict("Paris is in France.", ["??? !!!"])
    assert len(out_punct) == 1
    assert out_punct[0]["attention"] is not None
    assert len(out_punct[0]["attention"]) == 1
    assert out_punct[0]["attention"][0]["token"] == "<unk>"


def test_occlusion_check():
    """Occlusion check removes top-attended token and reports delta and stance shift."""
    ckpt_path = Path("ml/checkpoints/bilstm_attn.pt")
    if not ckpt_path.exists():
        pytest.skip("Checkpoint bilstm_attn.pt not found")

    vs = VerifierService(str(ckpt_path))
    assert vs.ready

    claim = "Paris is the capital of France."
    evidence = "Paris is the capital and largest city of France."
    res = vs.occlusion_check(claim, evidence)

    assert res is not None
    assert "top_token" in res
    assert "top_weight" in res
    assert "base_stance" in res
    assert "occluded_stance" in res
    assert "delta" in res
    assert res["base_stance"] == "SUPPORTED"
    assert isinstance(res["top_token"], str) and len(res["top_token"]) > 0
    assert res["top_weight"] > 0.0


def test_api_analyze_and_occlusion_endpoints():
    from fastapi.testclient import TestClient
    from backend.app.main import app

    client = TestClient(app)
    health = client.get("/api/health")
    assert health.status_code == 200

    # Test /api/analyze with weighted=True
    resp = client.post("/api/analyze", json={
        "text": "Paris is the capital of France.",
        "weighted": True,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["claims"]) == 1
    assert data["claims"][0]["verdict"] == "SUPPORTED"

    # Test /api/occlusion
    occ_resp = client.post("/api/occlusion", json={
        "claim": "Paris is the capital of France.",
        "evidence": "Paris is the capital of France."
    })
    assert occ_resp.status_code == 200
    occ_data = occ_resp.json()
    assert occ_data["base_stance"] == "SUPPORTED"
    assert "top_token" in occ_data
    assert "delta" in occ_data

