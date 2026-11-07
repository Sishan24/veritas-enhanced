import json
from pathlib import Path

VALID_VERDICTS = {"SUPPORTED", "REFUTED", "INSUFFICIENT"}
VALID_SOURCES = {"generator-true", "generator-false-variant", "synthetic-out-of-corpus"}


def _load_claims(path):
    p = Path(path)
    assert p.exists(), f"File does not exist: {path}"
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_claim_splits_load_and_validate():
    dev = _load_claims("data/claim_level/dev_claims.jsonl")
    test = _load_claims("data/claim_level/test_claims.jsonl")

    assert len(dev) >= 60, f"Expected at least 60 dev claims, got {len(dev)}"
    assert len(test) >= 60, f"Expected at least 60 test claims, got {len(test)}"

    for split_name, rows in [("dev", dev), ("test", test)]:
        claims = [r["claim"] for r in rows]
        assert len(claims) == len(set(claims)), f"Duplicate claims found in {split_name}"

        verdict_counts = {v: 0 for v in VALID_VERDICTS}
        for r in rows:
            assert "claim" in r and r["claim"].strip(), "Claim must be non-empty string"
            assert r["verdict"] in VALID_VERDICTS, f"Invalid verdict: {r['verdict']}"
            assert r["source"] in VALID_SOURCES, f"Invalid source: {r['source']}"
            verdict_counts[r["verdict"]] += 1

            ev_ids = r.get("justifying_evidence_ids")
            assert isinstance(ev_ids, list), "justifying_evidence_ids must be a list"

            if r["verdict"] in {"SUPPORTED", "REFUTED"}:
                assert len(ev_ids) >= 1, f"Expected non-empty evidence IDs for {r['verdict']}"
            elif r["verdict"] == "INSUFFICIENT":
                assert len(ev_ids) == 0, f"Expected empty evidence IDs for INSUFFICIENT"

        for v, count in verdict_counts.items():
            assert count >= 20, f"Quality gate failed: {split_name} has {count} claims for {v} (expected >= 20)"


def test_dev_and_test_are_disjoint():
    dev = _load_claims("data/claim_level/dev_claims.jsonl")
    test = _load_claims("data/claim_level/test_claims.jsonl")

    dev_claims = {r["claim"] for r in dev}
    test_claims = {r["claim"] for r in test}
    assert dev_claims.isdisjoint(test_claims), "Dev and test claims must be disjoint"

    dev_ev_ids = {ev for r in dev for ev in r["justifying_evidence_ids"]}
    test_ev_ids = {ev for r in test for ev in r["justifying_evidence_ids"]}
    assert dev_ev_ids.isdisjoint(test_ev_ids), f"Dev and test evidence IDs overlap: {dev_ev_ids & test_ev_ids}"
