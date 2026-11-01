"""Claim-evidence relation models: MLP, CNN, LSTM, BiLSTM, BiLSTM+Attention.

Every architecture encodes the claim and the evidence with a shared encoder, then a
classifier reads [c ; e ; |c-e| ; c*e] and predicts SUPPORTED / REFUTED / NOT_ENOUGH_INFO.
Only `bilstm_attn` weights evidence tokens by relevance to the claim (explainability).
"""
import torch
import torch.nn as nn

ARCHS = ["mlp", "cnn", "lstm", "bilstm", "bilstm_attn"]
ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "gelu": nn.GELU}


def masked_mean(x, mask):
    m = mask.unsqueeze(-1).float()
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


class MLPEncoder(nn.Module):
    def __init__(self, emb, hidden, act):
        super().__init__()
        self.out_dim = hidden
        self.fc = nn.Sequential(nn.Linear(emb, hidden), act())

    def forward(self, x, mask):
        return self.fc(masked_mean(x, mask)), None


class CNNEncoder(nn.Module):
    """Local n-gram features: parallel convolutions (k=2,3,4) + max-over-time pooling."""

    def __init__(self, emb, hidden, act, kernels=(2, 3, 4)):
        super().__init__()
        self.convs = nn.ModuleList(nn.Conv1d(emb, hidden // len(kernels) + 1, k, padding=k // 2) for k in kernels)
        self.act = act()
        self.out_dim = (hidden // len(kernels) + 1) * len(kernels)

    def forward(self, x, mask):
        h = x.transpose(1, 2)
        T = x.size(1)
        outs = [self.act(c(h))[:, :, :T] for c in self.convs]
        h = torch.cat(outs, dim=1).masked_fill(~mask.unsqueeze(1), -1e9)
        return h.max(dim=2).values, None


class RNNEncoder(nn.Module):
    def __init__(self, emb, hidden, bidirectional):
        super().__init__()
        self.rnn = nn.LSTM(emb, hidden, batch_first=True, bidirectional=bidirectional)
        self.out_dim = hidden * (2 if bidirectional else 1)

    def forward(self, x, mask):
        lengths = mask.sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        states, _ = nn.utils.rnn.pad_packed_sequence(
            out, batch_first=True, total_length=x.size(1)
        )
        return masked_mean(states, mask), states


class Verifier(nn.Module):
    def __init__(self, arch, vocab_size, emb_dim=100, hidden=128, dropout=0.3, activation="relu", pad_idx=0):
        super().__init__()
        assert arch in ARCHS, f"arch must be one of {ARCHS}"
        act = ACTIVATIONS[activation]
        self.arch = arch
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        if arch == "mlp":
            self.enc = MLPEncoder(emb_dim, hidden, act)
        elif arch == "cnn":
            self.enc = CNNEncoder(emb_dim, hidden, act)
        else:
            self.enc = RNNEncoder(emb_dim, hidden, bidirectional=arch in ("bilstm", "bilstm_attn"))
        d = self.enc.out_dim
        self.use_attn = arch == "bilstm_attn"
        if self.use_attn:
            self.attn_proj = nn.Linear(d, d, bias=False)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Sequential(nn.Linear(4 * d, hidden), act(), nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, claim, claim_mask, ev, ev_mask, return_attention=False):
        c, _ = self.enc(self.drop(self.embedding(claim)), claim_mask)
        e, states = self.enc(self.drop(self.embedding(ev)), ev_mask)
        attn = None
        if self.use_attn:
            scores = torch.bmm(self.attn_proj(states), c.unsqueeze(-1)).squeeze(-1)
            attn = torch.softmax(scores.masked_fill(~ev_mask, -1e9), dim=-1)
            e = torch.bmm(attn.unsqueeze(1), states).squeeze(1)
        feats = torch.cat([c, e, (c - e).abs(), c * e], dim=-1)
        logits = self.classifier(self.drop(feats))
        return (logits, attn) if return_attention else logits


def build_from_checkpoint(ckpt):
    cfg = ckpt["config"]
    model = Verifier(cfg["arch"], len(ckpt["vocab"]), cfg["emb_dim"], cfg["hidden"], cfg["dropout"], cfg["activation"])
    model.load_state_dict(ckpt["state"])
    return model.eval()
