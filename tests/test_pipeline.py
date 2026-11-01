import torch

from backend.app.services.claims import extract_claims
from backend.app.services.retriever import Retriever
from backend.app.services.verifier import decide
from ml.models import ARCHS, Verifier


def test_claim_extraction_skips_questions_and_fragments():
    claims = extract_claims("Paris is the capital of France. Is it big? Wow. The firm was founded in 1999.")
    assert [c["text"] for c in claims] == ["Paris is the capital of France.", "The firm was founded in 1999."]


def test_claim_extraction_journalism_verbs():
    """News-domain sentences should all be extracted as claims."""
    text = (
        "The UN Security Council convened an emergency session on Gaza. "
        "Spain won the UEFA Euro 2024 championship. "
        "Bitcoin surpassed 70 000 dollars in price. "
        "The WHO declared mpox a public health emergency."
    )
    claims = extract_claims(text)
    texts = [c["text"] for c in claims]
    assert len(claims) == 4, f"Expected 4 journalism claims, got {len(claims)}: {texts}"


def test_claim_extraction_rejects_disinfo_questions():
    """Conspiracy-style questions should not be extracted as claims."""
    claims = extract_claims("Did 5G towers cause COVID-19? Are chemtrails real?")
    assert claims == []


def test_claim_extraction_finds_entities():
    """Entities (proper nouns) should be captured in the entities field."""
    claims = extract_claims("Donald Trump was convicted of felony charges in New York.")
    assert claims
    entities = claims[0]["entities"]
    assert any("Trump" in e for e in entities)


def test_retriever_ranks_relevant_first():
    r = Retriever([
        {"id": "a", "text": "Cairo is the capital of Egypt."},
        {"id": "b", "text": "Bananas are yellow."},
    ])
    assert r.search("What is the capital of Egypt", 2)[0]["id"] == "a"


def test_retriever_journalism_passage():
    """Retriever should surface the relevant news passage over an unrelated one."""
    r = Retriever([
        {"id": "cop28", "text": "COP28 produced a text calling for a transition away from fossil fuels."},
        {"id": "sport", "text": "Real Madrid defeated Borussia Dortmund in the Champions League final."},
    ])
    hits = r.search("fossil fuel climate agreement COP28", 2)
    assert hits[0]["id"] == "cop28"


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


def test_verdict_disinformation_scenario():
    """A claim backed by strong refutation evidence should return REFUTED."""
    items = [_item(ref=0.92, score=0.45), _item(sup=0.1, ref=0.1, score=0.3)]
    verdict, conf = decide(items)
    assert verdict == "REFUTED"
    assert conf > 0.5


def test_verdict_insufficient_low_retrieval():
    """Low retrieval score should suppress verdict regardless of stance probability."""
    items = [_item(sup=0.95, score=0.02)]  # strong stance but irrelevant passage
    verdict, _ = decide(items)
    assert verdict == "INSUFFICIENT"
