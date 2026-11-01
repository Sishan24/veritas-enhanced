"""Claim extraction: sentence segmentation + claim-worthiness heuristics + entity-like spans.

Rule-based on purpose (no extra dependencies). Swap in spaCy NER or a trained claim-detector later.
"""
import re

_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\'(]?[A-Z0-9])')
_ENTITY = re.compile(r"(?<!^)(?<![.!?]\s)\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b")
_VERBS = {"is", "was", "are", "were", "has", "have", "had", "will", "said", "announced", "won", "founded",
          "born", "died", "caused", "increased", "decreased", "rose", "fell", "became", "contains", "causes"}


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
