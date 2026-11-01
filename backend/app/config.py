import os

from dotenv import load_dotenv

load_dotenv()

CHECKPOINT = os.getenv("VERITAS_CHECKPOINT", "ml/checkpoints/bilstm_attn.pt")
EVIDENCE_PATH = os.getenv("VERITAS_EVIDENCE_PATH", "data/sample/evidence.jsonl")
TOP_K = int(os.getenv("VERITAS_TOP_K", "3"))
CORS_ORIGINS = [o.strip() for o in os.getenv("VERITAS_CORS_ORIGINS", "http://localhost:5173").split(",")]
STANCE_THRESHOLD = 0.6   # min probability for an evidence passage to count as supporting/refuting
MIN_RETRIEVAL = 0.05     # below this TF-IDF score a passage is treated as irrelevant
