import torch

from backend.app.services.claims import extract_claims
from backend.app.services.retriever import Retriever
from backend.app.services.verifier import decide
from ml.models import ARCHS, Verifier


def test_claim_extraction_skips_questions_and_fragments():
    claims = extract_claims("Paris is the capital of France. Is it big? Wow. The firm was founded in 1999.")
    assert [c["text"] for c in claims] == ["Paris is the capital of France.", "The firm was founded in 1999."]


def test_retriever_ranks_relevant_first():
    r = Retriever([{"id": "a", "text": "Cairo is the capital of Egypt."}, {"id": "b", "text": "Bananas are yellow."}])
    assert r.search("What is the capital of Egypt", 2)[0]["id"] == "a"


def test_all_architectures_forward():
    c, e = torch.randint(2, 50, (4, 6)), torch.randint(2, 50, (4, 9))
    cm, em = torch.ones_like(c, dtype=torch.bool), torch.ones_like(e, dtype=torch.bool)
    for arch in ARCHS:
        out = Verifier(arch, 50)(c, cm, e, em)
        assert out.shape == (4, 3)


def _item(sup=0.0, ref=0.0, score=0.5):
    return {"retrieval_score": score, "probs": {"SUPPORTED": sup, "REFUTED": ref, "NOT_ENOUGH_INFO": 1 - sup - ref}}


def test_verdict_logic():
    assert decide([_item(sup=0.9)])[0] == "SUPPORTED"
    assert decide([_item(ref=0.9)])[0] == "REFUTED"
    assert decide([_item(sup=0.9), _item(ref=0.9)])[0] == "CONFLICTING"
    assert decide([_item(sup=0.9, score=0.0)])[0] == "INSUFFICIENT"
