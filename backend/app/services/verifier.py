"""Loads a trained checkpoint and scores (claim, evidence) pairs."""
from pathlib import Path

import torch

from backend.app.config import MIN_RETRIEVAL, STANCE_THRESHOLD
from ml.data import LABELS, Vocab, pad, tokenize
from ml.models import build_from_checkpoint

MAX_CLAIM, MAX_EV = 32, 64


class VerifierService:
    def __init__(self, checkpoint):
        self.path = Path(checkpoint)
        self.model = None
        self.arch = None
        if self.path.exists():
            ckpt = torch.load(self.path, map_location="cpu")
            self.model = build_from_checkpoint(ckpt)
            self.vocab = Vocab(ckpt["vocab"])
            self.arch = ckpt["config"]["arch"]

    @property
    def ready(self):
        return self.model is not None

    @torch.no_grad()
    def predict(self, claim, evidences):
        if not evidences:
            return []
        c = [self.vocab.encode(claim, MAX_CLAIM)] * len(evidences)
        e = [self.vocab.encode(x, MAX_EV) for x in evidences]
        ct, cm = pad(c)
        et, em = pad(e)
        logits, attn = self.model(ct, cm, et, em, return_attention=True)
        probs = torch.softmax(logits, -1)
        out = []
        for i, text in enumerate(evidences):
            p = {l: float(probs[i, j]) for j, l in enumerate(LABELS)}
            item = {"stance": max(p, key=p.get), "probs": p, "attention": None}
            if attn is not None:
                toks = tokenize(text)[:MAX_EV]
                item["attention"] = [{"token": t, "weight": float(attn[i, j])} for j, t in enumerate(toks)]
            out.append(item)
        return out


def decide(items, min_retrieval=MIN_RETRIEVAL, stance_threshold=STANCE_THRESHOLD):
    """Aggregate per-evidence stances into one of SUPPORTED / REFUTED / CONFLICTING / INSUFFICIENT.

    A claim with no relevant, confident evidence is INSUFFICIENT, never automatically false.
    """
    def top(item):
        return max(item["probs"], key=item["probs"].get)

    rel = [i for i in items if i["retrieval_score"] >= min_retrieval]
    sup = [i for i in rel if top(i) == "SUPPORTED" and i["probs"]["SUPPORTED"] >= stance_threshold]
    ref = [i for i in rel if top(i) == "REFUTED" and i["probs"]["REFUTED"] >= stance_threshold]
    if sup and ref:
        return "CONFLICTING", max(
            max(i["probs"]["SUPPORTED"] for i in sup),
            max(i["probs"]["REFUTED"] for i in ref),
        )
    if sup:
        return "SUPPORTED", max(i["probs"]["SUPPORTED"] for i in sup)
    if ref:
        return "REFUTED", max(i["probs"]["REFUTED"] for i in ref)
    return "INSUFFICIENT", max((i["probs"]["NOT_ENOUGH_INFO"] for i in rel), default=0.0)
