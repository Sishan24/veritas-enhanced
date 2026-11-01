"""TF-IDF evidence retrieval over a supplied passage collection."""
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

from backend.app.services.claims import split_sentences


def passages_from_docs(docs):
    """Split uploaded documents into sentence-level passages: [{id, text}]."""
    out = []
    for d in docs:
        for n, s in enumerate(split_sentences(d["text"]) or [d["text"]], 1):
            out.append({"id": f"{d['id']}#{n}", "text": s})
    return out


def load_passages(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


class Retriever:
    def __init__(self, passages):
        self.passages = passages
        self.vec = None
        if passages:
            self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
            self.matrix = self.vec.fit_transform([p["text"] for p in passages])

    def search(self, query, k=3):
        if self.vec is None:
            return []
        scores = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        order = scores.argsort()[::-1][:k]
        return [{**self.passages[i], "score": float(scores[i])} for i in order]
