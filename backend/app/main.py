import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.app import config
from backend.app.schemas import AnalyzeRequest
from backend.app.services.claims import extract_claims
from backend.app.services.retriever import Retriever, load_passages, passages_from_docs
from backend.app.services.verifier import VerifierService, decide

app = FastAPI(title="VERITAS", description="Claim & Context Intelligence Engine")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])

verifier = VerifierService(config.CHECKPOINT)
default_retriever = Retriever(load_passages(config.EVIDENCE_PATH))


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": verifier.ready, "arch": verifier.arch,
            "default_evidence": len(default_retriever.passages)}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not verifier.ready:
        raise HTTPException(503, f"No trained model at {config.CHECKPOINT}. Run: python -m ml.make_sample_data && python -m ml.train")
    k = req.top_k or config.TOP_K
    retriever = Retriever(passages_from_docs([d.model_dump() for d in req.evidence])) if req.evidence else default_retriever
    results = []
    for claim in extract_claims(req.text):
        hits = retriever.search(claim["text"], k)
        preds = verifier.predict(claim["text"], [h["text"] for h in hits])
        evidence = [{"id": h["id"], "text": h["text"], "retrieval_score": h["score"], **p} for h, p in zip(hits, preds)]
        verdict, confidence = decide(evidence)
        results.append({**claim, "verdict": verdict, "confidence": confidence, "evidence": evidence})
    return {"model": verifier.arch, "claims": results}


@app.get("/api/reports")
def reports():
    root = Path("reports")
    comparison = json.loads((root / "comparison.json").read_text()) if (root / "comparison.json").exists() else []
    runs = {}
    for h in root.glob("*/history.json"):
        m = h.parent / "metrics.json"
        runs[h.parent.name] = {"history": json.loads(h.read_text()), "metrics": json.loads(m.read_text()) if m.exists() else None}
    return {"comparison": comparison, "runs": runs}
