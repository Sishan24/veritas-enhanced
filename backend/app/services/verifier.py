"""Loads a trained checkpoint and scores (claim, evidence) pairs."""
from pathlib import Path
import re
import torch

from backend.app.config import MIN_RETRIEVAL, STANCE_THRESHOLD
from ml.data import DEFAULT_MAX_CLAIM, DEFAULT_MAX_EV, LABELS, Vocab, pad, tokenize
from ml.models import build_from_checkpoint

MAX_CLAIM = DEFAULT_MAX_CLAIM
MAX_EV = DEFAULT_MAX_EV


class VerifierService:
    def __init__(self, checkpoint):
        self.path = Path(checkpoint)
        self.model = None
        self.arch = None
        if self.path.exists():
            try:
                ckpt = torch.load(self.path, map_location="cpu", weights_only=False)
            except TypeError:
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
        claim_ids = self.vocab.encode(claim, MAX_CLAIM)
        c = [claim_ids] * len(evidences)
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
                if not toks:
                    toks = ["<unk>"]
                assert len(toks) == int(em[i].sum()), f"Tokens len ({len(toks)}) != encoded len ({int(em[i].sum())})"
                item["attention"] = [{"token": t, "weight": float(attn[i, j])} for j, t in enumerate(toks)]
            out.append(item)
        return out

    @torch.no_grad()
    def occlusion_check(self, claim, evidence):
        """Explainability occlusion test: remove the top-attended token to test if stance moves."""
        if not self.ready:
            return None
        res = self.predict(claim, [evidence])
        if not res or not res[0].get("attention"):
            return None
        item = res[0]
        base_stance = item["stance"]
        base_prob = item["probs"][base_stance]
        attn = item["attention"]
        if not attn:
            return None
        top_tok_entry = max(attn, key=lambda a: a["weight"])
        top_token = top_tok_entry["token"]

        pattern = re.compile(rf"\b{re.escape(top_token)}\b", re.IGNORECASE)
        occluded_text = pattern.sub("", evidence).strip()
        if not occluded_text:
            occluded_text = "<unk>"

        occ_res = self.predict(claim, [occluded_text])
        if not occ_res:
            return None
        occ_item = occ_res[0]
        occ_stance = occ_item["stance"]
        occ_prob = occ_item["probs"].get(base_stance, 0.0)
        delta = base_prob - occ_prob

        return {
            "top_token": top_token,
            "top_weight": top_tok_entry["weight"],
            "base_stance": base_stance,
            "base_prob": base_prob,
            "occluded_stance": occ_stance,
            "occluded_prob": occ_prob,
            "delta": delta,
            "flipped": base_stance != occ_stance,
            "occluded_text": occluded_text,
        }


def decide(items, min_retrieval=MIN_RETRIEVAL, stance_threshold=STANCE_THRESHOLD, weighted=False):
    """Aggregate per-evidence stances into one of SUPPORTED / REFUTED / CONFLICTING / INSUFFICIENT.

    A claim with no relevant, confident evidence is INSUFFICIENT, never automatically false.
    When weighted=True, votes and confidence are weighted by retrieval_score to prevent weak
    marginal hits from disproportionately flipping decisions.
    """
    def top(item):
        return max(item["probs"], key=item["probs"].get)

    rel = [i for i in items if i["retrieval_score"] >= min_retrieval]
    sup = [i for i in rel if top(i) == "SUPPORTED" and i["probs"]["SUPPORTED"] >= stance_threshold]
    ref = [i for i in rel if top(i) == "REFUTED" and i["probs"]["REFUTED"] >= stance_threshold]

    if not weighted:
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

    # Retrieval-weighted aggregation
    if sup and ref:
        sup_weight = sum(i["retrieval_score"] for i in sup)
        ref_weight = sum(i["retrieval_score"] for i in ref)
        total = sup_weight + ref_weight
        if sup_weight / total >= 0.80:
            weighted_conf = sum(i["probs"]["SUPPORTED"] * i["retrieval_score"] for i in sup) / sup_weight
            return "SUPPORTED", weighted_conf
        if ref_weight / total >= 0.80:
            weighted_conf = sum(i["probs"]["REFUTED"] * i["retrieval_score"] for i in ref) / ref_weight
            return "REFUTED", weighted_conf
        weighted_conf = (
            sum(i["probs"]["SUPPORTED"] * i["retrieval_score"] for i in sup) +
            sum(i["probs"]["REFUTED"] * i["retrieval_score"] for i in ref)
        ) / total
        return "CONFLICTING", weighted_conf

    if sup:
        sup_weight = sum(i["retrieval_score"] for i in sup)
        weighted_conf = sum(i["probs"]["SUPPORTED"] * i["retrieval_score"] for i in sup) / (sup_weight or 1.0)
        return "SUPPORTED", weighted_conf

    if ref:
        ref_weight = sum(i["retrieval_score"] for i in ref)
        weighted_conf = sum(i["probs"]["REFUTED"] * i["retrieval_score"] for i in ref) / (ref_weight or 1.0)
        return "REFUTED", weighted_conf

    nei_scores = [(i["probs"]["NOT_ENOUGH_INFO"], i["retrieval_score"]) for i in rel]
    if not nei_scores:
        return "INSUFFICIENT", 0.0
    tot_w = sum(w for _, w in nei_scores)
    weighted_conf = sum(p * w for p, w in nei_scores) / (tot_w or 1.0)
    return "INSUFFICIENT", weighted_conf
