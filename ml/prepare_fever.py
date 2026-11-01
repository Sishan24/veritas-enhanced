"""Convert FEVER into the (claim, evidence, label) rows used by ml.train.

Download first (see data/README.md):  train.jsonl, shared_task_dev.jsonl and the wiki-pages/ shards.
  python -m ml.prepare_fever --claims data/raw/train.jsonl --wiki data/raw/wiki-pages --out data/processed/train.jsonl --limit 50000

SUPPORTS/REFUTES use the annotated evidence sentence. NOT ENOUGH INFO has no gold evidence, so we pair the
claim with a random sentence from another claim (a weak baseline). For a stronger setup, replace this with
retrieved sentences (see backend/app/services/retriever.py).
"""
import argparse
import json
import random
from pathlib import Path

MAP = {"SUPPORTS": "SUPPORTED", "REFUTES": "REFUTED", "NOT ENOUGH INFO": "NOT_ENOUGH_INFO"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims", required=True)
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=50000)
    a = ap.parse_args()
    rng = random.Random(0)

    claims = [json.loads(l) for l in open(a.claims, encoding="utf-8")][: a.limit]
    need = set()
    for c in claims:
        for group in c["evidence"]:
            for _, _, page, sid in group:
                if page is not None:
                    need.add((page, sid))

    sents = {}
    for shard in sorted(Path(a.wiki).glob("*.jsonl")):
        for line in open(shard, encoding="utf-8"):
            d = json.loads(line)
            for ln in d["lines"].split("\n"):
                parts = ln.split("\t")
                if len(parts) > 1 and parts[0].isdigit() and (d["id"], int(parts[0])) in need:
                    sents[(d["id"], int(parts[0]))] = parts[1]

    rows, pool = [], []
    for c in claims:
        if c["label"] == "NOT ENOUGH INFO":
            continue
        for _, _, page, sid in c["evidence"][0]:
            if (page, sid) in sents:
                rows.append({"claim": c["claim"], "evidence": sents[(page, sid)], "label": MAP[c["label"]]})
                pool.append(sents[(page, sid)])
                break
    for c in claims:
        if c["label"] == "NOT ENOUGH INFO" and pool:
            rows.append({"claim": c["claim"], "evidence": rng.choice(pool), "label": "NOT_ENOUGH_INFO"})

    rng.shuffle(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(f"Wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
