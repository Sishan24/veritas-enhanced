import json
import re
from collections import Counter

import torch
from torch.utils.data import Dataset

LABELS = ["SUPPORTED", "REFUTED", "NOT_ENOUGH_INFO"]
L2I = {l: i for i, l in enumerate(LABELS)}
PAD, UNK = 0, 1
DEFAULT_MAX_CLAIM = 32
DEFAULT_MAX_EV = 64
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class Vocab:
    def __init__(self, itos):
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}

    @classmethod
    def build(cls, texts, min_freq=1, max_size=30000):
        counts = Counter(t for x in texts for t in tokenize(x))
        words = [w for w, c in counts.most_common(max_size) if c >= min_freq]
        return cls(["<pad>", "<unk>"] + words)

    def encode(self, text, max_len):
        ids = [self.stoi.get(t, UNK) for t in tokenize(text)][:max_len]
        return ids or [UNK]

    def __len__(self):
        return len(self.itos)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class ClaimEvidenceDataset(Dataset):
    """Rows: {"claim": str, "evidence": str, "label": SUPPORTED|REFUTED|NOT_ENOUGH_INFO}"""

    def __init__(self, path, vocab, max_claim=DEFAULT_MAX_CLAIM, max_ev=DEFAULT_MAX_EV):
        self.rows = read_jsonl(path)
        self.vocab, self.mc, self.me = vocab, max_claim, max_ev

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.vocab.encode(r["claim"], self.mc), self.vocab.encode(r["evidence"], self.me), L2I[r["label"]]


def pad(seqs):
    n = max(len(s) for s in seqs)
    t = torch.zeros(len(seqs), n, dtype=torch.long)
    for i, s in enumerate(seqs):
        t[i, : len(s)] = torch.tensor(s)
    return t, t != PAD


def collate(batch):
    claims, evs, ys = zip(*batch)
    c, cm = pad(claims)
    e, em = pad(evs)
    return c, cm, e, em, torch.tensor(ys)
