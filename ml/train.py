"""Train one claim-evidence model.

  python -m ml.train --arch bilstm_attn --epochs 12
  python -m ml.train --arch cnn --optimizer nesterov --activation tanh
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

from ml.data import LABELS, ClaimEvidenceDataset, Vocab, collate, read_jsonl
from ml.models import ACTIVATIONS, ARCHS, Verifier

OPTIMIZERS = ["sgd", "momentum", "nesterov", "adam"]


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=ARCHS, default="bilstm_attn")
    p.add_argument("--data-dir", default="data/sample")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 for adam, 5e-2 for SGD variants")
    p.add_argument("--activation", choices=list(ACTIVATIONS), default="relu")
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--patience", type=int, default=4, help="early stopping on dev macro-F1; 0 disables")
    p.add_argument("--emb-dim", type=int, default=100)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--name", default=None, help="run name (default: arch)")
    p.add_argument("--no-save", action="store_true", help="skip writing the checkpoint")
    p.add_argument("--split", choices=["pair", "passage"], default="pair",
                   help="Data split mode: pair=data/sample (default); passage=data/sample_passage.")
    return p.parse_args(argv)


def make_optimizer(name, params, lr, wd):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr or 1e-3, weight_decay=wd)
    lr = lr or 5e-2
    mom = {"sgd": 0.0, "momentum": 0.9, "nesterov": 0.9}[name]
    return torch.optim.SGD(params, lr=lr, momentum=mom, nesterov=name == "nesterov", weight_decay=wd)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    ys, ps, loss, secs = [], [], 0.0, 0.0
    for c, cm, e, em, y in loader:
        c, cm, e, em, y = (t.to(device) for t in (c, cm, e, em, y))
        t0 = time.perf_counter()
        logits = model(c, cm, e, em)
        secs += time.perf_counter() - t0
        loss += crit(logits, y).item()
        ys += y.tolist()
        ps += logits.argmax(-1).tolist()
    pr, rc, f1, _ = precision_recall_fscore_support(ys, ps, average="macro", zero_division=0)
    return {
        "loss": loss / len(ys), "accuracy": accuracy_score(ys, ps), "macro_f1": f1, "precision": pr, "recall": rc,
        "confusion_matrix": confusion_matrix(ys, ps, labels=range(len(LABELS))).tolist(),
        "latency_ms": 1000 * secs / len(ys),
    }


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    split = getattr(args, "split", "pair")
    # Resolve data directory: explicit --data-dir overrides defaults
    if args.data_dir != "data/sample":
        d = Path(args.data_dir)
    elif split == "passage":
        d = Path("data/sample_passage")
    else:
        d = Path("data/sample")
    train_rows = read_jsonl(d / "train.jsonl")
    min_freq = 2 if split == "passage" else 1
    vocab = Vocab.build([r["claim"] for r in train_rows] + [r["evidence"] for r in train_rows],
                        min_freq=min_freq)
    mk = lambda f, sh: DataLoader(ClaimEvidenceDataset(d / f, vocab), args.batch_size, shuffle=sh, collate_fn=collate)
    train, dev, test = mk("train.jsonl", True), mk("dev.jsonl", False), mk("test.jsonl", False)

    model = Verifier(args.arch, len(vocab), args.emb_dim, args.hidden, args.dropout, args.activation).to(device)
    opt = make_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    crit = nn.CrossEntropyLoss()
    split_tag = f"_{split}" if split == "passage" else ""
    name = args.name or f"{args.arch}{split_tag}_s{args.seed}"
    history, best, best_state, bad = [], -1.0, None, 0
    t_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for c, cm, e, em, y in train:
            c, cm, e, em, y = (t.to(device) for t in (c, cm, e, em, y))
            opt.zero_grad()
            loss = crit(model(c, cm, e, em), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * len(y)
        m = evaluate(model, dev, device)
        history.append({"epoch": epoch, "train_loss": total / len(train.dataset), "dev_loss": m["loss"],
                        "dev_accuracy": m["accuracy"], "dev_macro_f1": m["macro_f1"]})
        print(f"[{name}] epoch {epoch:02d} train {history[-1]['train_loss']:.4f} dev_f1 {m['macro_f1']:.4f}")
        if m["macro_f1"] > best:
            best, bad = m["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"[{name}] early stop at epoch {epoch}")
                break

    train_seconds = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    test_m = evaluate(model, test, device)
    result = {"name": name, "arch": args.arch, "seed": args.seed,
              "n_params": sum(p.numel() for p in model.parameters()),
              "optimizer": args.optimizer, "activation": args.activation,
              "dropout": args.dropout, "weight_decay": args.weight_decay, "epochs_run": len(history),
              "train_seconds": train_seconds, "best_dev_macro_f1": best, **test_m, "labels": LABELS}

    out = Path("reports") / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "history.json").write_text(json.dumps(history, indent=2))
    if not args.no_save:
        Path("ml/checkpoints").mkdir(parents=True, exist_ok=True)
        cfg = {"arch": args.arch, "emb_dim": args.emb_dim, "hidden": args.hidden, "dropout": args.dropout, "activation": args.activation}
        torch.save({"config": cfg, "state": best_state, "vocab": vocab.itos}, f"ml/checkpoints/{name}.pt")
    print(f"[{name}] test acc {test_m['accuracy']:.4f} macro-F1 {test_m['macro_f1']:.4f} latency {test_m['latency_ms']:.3f} ms/pair")
    return result


if __name__ == "__main__":
    run(parse())
