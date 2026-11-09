"""Demo preparation and verification script using FastAPI TestClient through the real backend API path."""
import hashlib
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from backend.app import config
from backend.app.main import app


def compute_sha256(path_str: str) -> str:
    p = Path(path_str)
    if not p.exists():
        return f"FILE_NOT_FOUND ({path_str})"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


CANDIDATE_CLAIMS = [
    # Dev set (10 claims: 4 SUP, 4 REF, 2 INS)
    {"text": "Paris is the capital of France.", "origin": "dev-SUPPORTED"},
    {"text": "Sirius Corporation was founded in 2004.", "origin": "dev-SUPPORTED"},
    {"text": "2023 was confirmed as the hottest year on record globally.", "origin": "dev-SUPPORTED"},
    {"text": "Bangkok is the capital of Thailand.", "origin": "dev-SUPPORTED"},
    {"text": "Marseille is the capital of France.", "origin": "dev-REFUTED"},
    {"text": "Sirius Corporation was founded in 2016.", "origin": "dev-REFUTED"},
    {"text": "Chiang Mai is the capital of Thailand.", "origin": "dev-REFUTED"},
    {"text": "Istanbul is the capital of Turkey.", "origin": "dev-REFUTED"},
    {"text": "Paris banned all private passenger vehicles from its city center on weekdays.", "origin": "dev-INSUFFICIENT"},
    {"text": "Bangkok completed an electric ferry transit project along the Chao Phraya River.", "origin": "dev-INSUFFICIENT"},

    # Test set (10 claims: 4 SUP, 4 REF, 2 INS)
    {"text": "Madrid is the capital of Spain.", "origin": "test-SUPPORTED"},
    {"text": "Acme Corporation was founded in 1948.", "origin": "test-SUPPORTED"},
    {"text": "Apple reached a market capitalisation of three trillion dollars.", "origin": "test-SUPPORTED"},
    {"text": "Stockholm is the capital of Sweden.", "origin": "test-SUPPORTED"},
    {"text": "Barcelona is the capital of Spain.", "origin": "test-REFUTED"},
    {"text": "Acme Corporation was founded in 1945.", "origin": "test-REFUTED"},
    {"text": "Microsoft became the first company to reach a three trillion dollar market capitalisation.", "origin": "test-REFUTED"},
    {"text": "Gothenburg is the capital of Sweden.", "origin": "test-REFUTED"},
    {"text": "Madrid completed an expansion of its underground thermal heating grid across forty districts.", "origin": "test-INSUFFICIENT"},
    {"text": "Stockholm converted its municipal public bus fleet entirely to hydrogen fuel cells.", "origin": "test-INSUFFICIENT"},

    # Corpus topics (10 claims: 4 SUP, 4 REF, 2 Out-of-corpus)
    {"text": "The US Federal Reserve raised interest rates to a 22-year high.", "origin": "corpus-SUPPORTED"},
    {"text": "Spain won the UEFA Euro 2024 championship.", "origin": "corpus-SUPPORTED"},
    {"text": "Canada experienced its worst wildfire season on record in 2023.", "origin": "corpus-SUPPORTED"},
    {"text": "Global CO2 emissions from fossil fuels reached a new record in 2023.", "origin": "corpus-SUPPORTED"},
    {"text": "The US Federal Reserve lowered interest rates to record lows in 2023.", "origin": "corpus-REFUTED"},
    {"text": "England won the UEFA Euro 2024 championship in Berlin.", "origin": "corpus-REFUTED"},
    {"text": "Canada had its mildest wildfire season in modern history in 2023.", "origin": "corpus-REFUTED"},
    {"text": "Global CO2 emissions from fossil fuels decreased significantly in 2023.", "origin": "corpus-REFUTED"},
    {"text": "The European Space Agency launched a rover to explore Venus in 2024.", "origin": "corpus-OUT-OF-CORPUS"},
    {"text": "The international committee approved new guidelines for lunar mining rights.", "origin": "corpus-OUT-OF-CORPUS"},
]

CONFLICTING_ATTEMPTS = [
    {
        "attempt": 1,
        "claim": "Ankara is the capital of Turkey.",
        "evidence": [
            {"id": "user-sup-1", "text": "The capital city of Turkey is Ankara."},
            {"id": "user-ref-1", "text": "The capital city of Turkey is Istanbul, according to historical maritime records."},
        ],
    },
    {
        "attempt": 2,
        "claim": "Acme Corporation was founded in 1948.",
        "evidence": [
            {"id": "user-sup-2", "text": "Acme Corporation was established in 1948 by three partners."},
            {"id": "user-ref-2", "text": "Official corporate archives verify that Acme Corporation was founded in 1975, not in 1948."},
        ],
    },
    {
        "attempt": 3,
        "claim": "The Federal Reserve raised interest rates.",
        "evidence": [
            {"id": "user-sup-3", "text": "The Federal Reserve raised benchmark interest rates to a multi-decade high."},
            {"id": "user-ref-3", "text": "The Federal Reserve lowered benchmark interest rates by fifty basis points."},
        ],
    },
]


def run_demo_check():
    ckpt_hash = compute_sha256(config.CHECKPOINT)
    corpus_hash = compute_sha256(config.EVIDENCE_PATH)

    lines = []
    lines.append("================================================================================")
    lines.append("VERITAS DEMO PREPARATION & VERIFICATION REPORT")
    lines.append("================================================================================")
    lines.append(f"MIN_RETRIEVAL:      {config.MIN_RETRIEVAL}")
    lines.append(f"STANCE_THRESHOLD:   {config.STANCE_THRESHOLD}")
    lines.append(f"TOP_K:              {config.TOP_K}")
    lines.append(f"Deployed Checkpoint: {config.CHECKPOINT}")
    lines.append(f"Checkpoint SHA-256: {ckpt_hash}")
    lines.append(f"Evidence Corpus:    {config.EVIDENCE_PATH}")
    lines.append(f"Corpus SHA-256:     {corpus_hash}")
    lines.append("================================================================================\n")

    client = TestClient(app)
    health = client.get("/api/health").json()
    lines.append(f"API Health Check: status={health.get('status')}, model_ready={health.get('model_ready')}, arch={health.get('arch')}, passages={health.get('default_evidence')}\n")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("PART A: 30 CANDIDATE CLAIMS RUN 3 TIMES THROUGH /api/analyze")
    lines.append("Selection method: 10 dev claims (4 SUP, 4 REF, 2 INS), 10 test claims (4 SUP, 4 REF, 2 INS), and 10 corpus-topic claims (4 SUP, 4 REF, 2 out-of-corpus).")
    lines.append("--------------------------------------------------------------------------------\n")

    evaluated_candidates = []

    for idx, cand in enumerate(CANDIDATE_CLAIMS, 1):
        runs = []
        for r_num in range(3):
            resp = client.post("/api/analyze", json={"text": cand["text"]})
            data = resp.json()
            claims_out = data.get("claims", [])
            runs.append(claims_out)

        # Check stability across runs
        extracted_counts = [len(r) for r in runs]
        same_extraction = (len(set(extracted_counts)) == 1)
        single_claim = (extracted_counts[0] == 1)

        first_claim = runs[0][0] if runs[0] else None
        verdicts = [r[0]["verdict"] for r in runs if r]
        confidences = [r[0]["confidence"] for r in runs if r]
        evidence_ids = [[e["id"] for e in r[0]["evidence"]] for r in runs if r]

        identical_verdicts = (len(set(verdicts)) == 1)
        mean_conf = sum(confidences) / len(confidences) if confidences else 0.0

        # Safety criteria:
        # exactly one claim extracted, verdict identical on all 3 runs, abs(confidence - 0.6) >= 0.05
        is_safe = single_claim and identical_verdicts and (abs(mean_conf - 0.60) >= 0.05)

        cand_result = {
            "index": idx,
            "text": cand["text"],
            "origin": cand["origin"],
            "single_claim": single_claim,
            "extracted_count": extracted_counts[0] if single_claim else extracted_counts,
            "identical_verdicts": identical_verdicts,
            "verdict": verdicts[0] if identical_verdicts else "VARIED",
            "confidence": mean_conf,
            "conf_safe_margin": abs(mean_conf - 0.60) >= 0.05,
            "evidence_ids": evidence_ids[0] if evidence_ids else [],
            "is_safe": is_safe,
        }
        evaluated_candidates.append(cand_result)

        safe_tag = "[DEMO-SAFE]" if is_safe else "[NOT-SAFE]"
        lines.append(f"[{idx:02d}] {safe_tag} Origin: {cand['origin']}")
        lines.append(f"     Claim: \"{cand['text']}\"")
        lines.append(f"     Extracted: {extracted_counts} claims | Verdict: {verdicts[0] if identical_verdicts else verdicts} | Conf: {mean_conf:.4f} (|conf-0.6|={abs(mean_conf-0.6):.4f})")
        lines.append(f"     Evidence IDs: {evidence_ids[0] if evidence_ids else []}")
        if not is_safe:
            reasons = []
            if not single_claim:
                reasons.append(f"Extracted count != 1 ({extracted_counts})")
            if not identical_verdicts:
                reasons.append("Verdicts differ across runs")
            if abs(mean_conf - 0.60) < 0.05:
                reasons.append(f"Confidence {mean_conf:.4f} within 0.05 margin of 0.60")
            lines.append(f"     Unsafe reason: {', '.join(reasons)}")
        lines.append("")

    safe_supported = [c for c in evaluated_candidates if c["is_safe"] and c["verdict"] == "SUPPORTED"]
    safe_refuted = [c for c in evaluated_candidates if c["is_safe"] and c["verdict"] == "REFUTED"]
    safe_insufficient = [c for c in evaluated_candidates if c["is_safe"] and c["verdict"] == "INSUFFICIENT"]

    lines.append("--------------------------------------------------------------------------------")
    lines.append("DEMO SAFETY SUMMARY (Deployed Model: bilstm_attn.pt)")
    lines.append("--------------------------------------------------------------------------------")
    lines.append(f"Total candidate claims tested: {len(evaluated_candidates)}")
    lines.append(f"Total DEMO-SAFE claims: {sum(c['is_safe'] for c in evaluated_candidates)}")
    lines.append(f"  - Safe SUPPORTED claims:    {len(safe_supported)}")
    lines.append(f"  - Safe REFUTED claims:      {len(safe_refuted)}")
    lines.append(f"  - Safe INSUFFICIENT claims: {len(safe_insufficient)}")
    lines.append("")
    if not safe_refuted:
        lines.append("PLAIN STATEMENT: No REFUTED claims qualified as demo-safe under the deployed checkpoint.")
    else:
        lines.append(f"Found {len(safe_refuted)} demo-safe REFUTED claims.")
    lines.append("")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("PART B: CONFLICTING VERDICT ATTEMPTS (User-Supplied Evidence)")
    lines.append("Note: Labelled 'demo-only user-supplied evidence'. Corpus was NOT modified.")
    lines.append("--------------------------------------------------------------------------------\n")

    conflicting_results = []
    for att in CONFLICTING_ATTEMPTS:
        att_num = att["attempt"]
        claim_text = att["claim"]
        ev_docs = att["evidence"]

        resp = client.post("/api/analyze", json={"text": claim_text, "evidence": ev_docs})
        data = resp.json()
        claims_out = data.get("claims", [])
        verdict = claims_out[0]["verdict"] if claims_out else "NO_CLAIM"
        conf = claims_out[0]["confidence"] if claims_out else 0.0
        ev_breakdown = claims_out[0].get("evidence", []) if claims_out else []

        conf_res = {
            "attempt": att_num,
            "claim": claim_text,
            "verdict": verdict,
            "confidence": conf,
            "evidence": ev_breakdown,
        }
        conflicting_results.append(conf_res)

        lines.append(f"Attempt {att_num} [demo-only user-supplied evidence]:")
        lines.append(f"  Claim: \"{claim_text}\"")
        lines.append(f"  Supplied Passages:")
        for doc in ev_docs:
            lines.append(f"    - [{doc['id']}]: {doc['text']}")
        lines.append(f"  Resulting Verdict: {verdict} (Confidence: {conf:.4f})")
        lines.append(f"  Evidence Item Stances:")
        for item in ev_breakdown:
            lines.append(f"    - id={item.get('id')} ret_score={item.get('retrieval_score', 0):.4f} probs={item.get('probs')}")
        lines.append("")

    successes = [r for r in conflicting_results if r["verdict"] == "CONFLICTING"]
    if not successes:
        lines.append("PLAIN STATEMENT: CONFLICTING verdict cannot be shown reliably through user-supplied evidence.")
    else:
        lines.append(f"CONFLICTING verdict achieved on {len(successes)} of 3 attempts.")

    full_output = "\n".join(lines)
    print(full_output)

    out_path = Path("results/demo_check_output.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_output, encoding="utf-8")
    print(f"\nWrote demo check log: {out_path}")

    return {
        "candidates": evaluated_candidates,
        "safe_supported": safe_supported,
        "safe_refuted": safe_refuted,
        "safe_insufficient": safe_insufficient,
        "conflicting_results": conflicting_results,
    }


if __name__ == "__main__":
    run_demo_check()
