"""Claim extraction: sentence segmentation + claim-worthiness heuristics + entity-like spans.

Rule-based on purpose (no extra dependencies). Swap in spaCy NER or a trained claim-detector later.
"""
import re

_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\'(]?[A-Z0-9])')
_ENTITY = re.compile(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")
_VERBS = {
    # copula / auxiliary
    "is", "was", "are", "were", "has", "have", "had", "will", "be", "been",
    # factual reporting verbs
    "said", "announced", "confirmed", "stated", "claimed", "denied", "alleged",
    "reported", "revealed", "disclosed", "declared", "admitted", "conceded",
    "warned", "urged", "demanded", "called", "proposed", "recommended",
    "accused", "charged", "convicted", "acquitted", "sentenced", "arrested",
    "indicted", "prosecuted", "sued", "settled", "ruled", "overturned", "upheld",
    # events and change
    "won", "lost", "defeated", "beat", "signed", "passed", "rejected", "vetoed",
    "approved", "adopted", "enacted", "repealed", "launched", "began", "started",
    "ended", "halted", "suspended", "resumed", "cancelled", "postponed", "delayed",
    "deployed", "withdrew", "invaded", "attacked", "bombed", "killed", "died",
    "injured", "displaced", "fled", "surrendered", "seized", "captured",
    # economics / statistics
    "increased", "decreased", "rose", "fell", "surged", "plunged", "dropped",
    "climbed", "slipped", "grew", "contracted", "expanded", "shrank", "reached",
    "exceeded", "crossed", "hit", "missed", "cut", "raised", "pledged", "invested",
    "allocated", "divested", "borrowed", "defaulted", "sanctioned", "imposed",
    # founding / attribution
    "founded", "established", "created", "formed", "merged", "acquired", "sold",
    "spun", "dissolved", "incorporated", "privatised", "nationalised",
    # life events
    "born", "graduated", "elected", "appointed", "resigned", "retired", "fired",
    "promoted", "succeeded", "replaced", "named", "awarded", "honoured",
    # causation / relationship
    "caused", "triggered", "led", "resulted", "linked", "associated", "connected",
    "attributed", "blamed", "credited", "contains", "causes", "became", "remained",
}


def split_sentences(text):
    text = re.sub(r"\s+", " ", text.strip())
    return [s.strip() for s in _SPLIT.split(text) if s.strip()]


def is_claim(sentence):
    words = sentence.split()
    if not 4 <= len(words) <= 60 or sentence.endswith("?"):
        return False
    lower = {w.lower().strip(".,;:!\"'()") for w in words}
    return bool(lower & _VERBS) or any(ch.isdigit() for ch in sentence)


def extract_claims(text):
    claims = []
    for s in split_sentences(text):
        if is_claim(s):
            ents = sorted({e for e in _ENTITY.findall(s)})
            claims.append({"id": f"c{len(claims) + 1}", "text": s, "entities": ents,
                           "has_number": any(ch.isdigit() for ch in s)})
    return claims
