"""Baseline classifiers for VERITAS claim-verification.

Trains two baselines on the SAME train/test split the neural trainer uses
(data/sample/ for pair split, data/sample_passage/ for passage split):

  1. majority  – sklearn DummyClassifier(strategy="most_frequent")
  2. tfidf_logreg – TfidfVectorizer (claim) + TfidfVectorizer (evidence),
                   features hstacked, then LogisticRegression(max_iter=1000)

Output mirrors the neural metrics.json schema exactly so the Neural Lab
screen renders both baselines without any frontend change.

Usage:
    python -m ml.baselines                    # pair split (default)
    python -m ml.baselines --split passage    # passage split
"""
import argparse
import json
import time
from pathlib import Path
from scipy.sparse import hstack

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from ml.data import LABELS, L2I, read_jsonl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data_dir(split: str) -> Path:
    return Path("data/sample_passage") if split == "passage" else Path("data/sample")


def _load_split(data_dir: Path, filename: str):
    rows = read_jsonl(data_dir / filename)
    claims = [r["claim"] for r in rows]
    evidences = [r["evidence"] for r in rows]
    labels = [L2I[r["label"]] for r in rows]
    return claims, evidences, labels


def _metrics_dict(
    name: str,
    arch_tag: str,
    seed: int,
    ys: list,
    ps: list,
    train_seconds: float,
    best_dev_macro_f1: float,
    latency_ms: float,
) -> dict:
    """Build a metrics dict with the same keys as a neural run's metrics.json."""
    pr, rc, f1, _ = precision_recall_fscore_support(
        ys, ps, average="macro", zero_division=0
    )
    cm = confusion_matrix(ys, ps, labels=list(range(len(LABELS)))).tolist()
    return {
        "name": name,
        "arch": arch_tag,
        "seed": seed,
        "n_params": 0,
        "optimizer": None,
        "activation": None,
        "dropout": None,
        "weight_decay": None,
        "epochs_run": 0,
        "train_seconds": train_seconds,
        "best_dev_macro_f1": best_dev_macro_f1,
        "loss": None,
        "accuracy": accuracy_score(ys, ps),
        "macro_f1": float(f1),
        "precision": float(pr),
        "recall": float(rc),
        "confusion_matrix": cm,
        "latency_ms": latency_ms,
        "labels": LABELS,
    }


def _write_report(out_dir: Path, result: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Baseline 1: majority class
# ---------------------------------------------------------------------------

def run_majority(
    train_claims, train_evs, train_labels,
    dev_claims, dev_evs, dev_labels,
    test_claims, test_evs, test_labels,
    name: str,
    seed: int = 0,
) -> dict:
    clf = DummyClassifier(strategy="most_frequent", random_state=seed)
    t0 = time.perf_counter()
    clf.fit(train_claims, train_labels)
    train_seconds = time.perf_counter() - t0

    # dev best_dev_macro_f1
    dev_ps = clf.predict(dev_claims)
    _, _, dev_f1, _ = precision_recall_fscore_support(
        dev_labels, dev_ps, average="macro", zero_division=0
    )

    # test inference timing
    t1 = time.perf_counter()
    test_ps = clf.predict(test_claims).tolist()
    latency_ms = 1000 * (time.perf_counter() - t1) / max(len(test_labels), 1)

    return _metrics_dict(
        name=name,
        arch_tag="majority",
        seed=seed,
        ys=test_labels,
        ps=test_ps,
        train_seconds=train_seconds,
        best_dev_macro_f1=float(dev_f1),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Baseline 2: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------

def run_tfidf_logreg(
    train_claims, train_evs, train_labels,
    dev_claims, dev_evs, dev_labels,
    test_claims, test_evs, test_labels,
    name: str,
    seed: int = 0,
) -> dict:
    vec_c = TfidfVectorizer(sublinear_tf=True)
    vec_e = TfidfVectorizer(sublinear_tf=True)

    t0 = time.perf_counter()
    X_train = hstack([vec_c.fit_transform(train_claims), vec_e.fit_transform(train_evs)])
    X_dev   = hstack([vec_c.transform(dev_claims),   vec_e.transform(dev_evs)])
    X_test  = hstack([vec_c.transform(test_claims),  vec_e.transform(test_evs)])

    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
    clf.fit(X_train, train_labels)
    train_seconds = time.perf_counter() - t0

    # dev best_dev_macro_f1
    dev_ps = clf.predict(X_dev)
    _, _, dev_f1, _ = precision_recall_fscore_support(
        dev_labels, dev_ps, average="macro", zero_division=0
    )

    # test inference timing
    t1 = time.perf_counter()
    test_ps = clf.predict(X_test).tolist()
    latency_ms = 1000 * (time.perf_counter() - t1) / max(len(test_labels), 1)

    return _metrics_dict(
        name=name,
        arch_tag="tfidf_logreg",
        seed=seed,
        ys=test_labels,
        ps=test_ps,
        train_seconds=train_seconds,
        best_dev_macro_f1=float(dev_f1),
        latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse(argv=None):
    p = argparse.ArgumentParser(description="Train baseline classifiers for VERITAS.")
    p.add_argument(
        "--split",
        choices=["pair", "passage"],
        default="pair",
        help="Which data split to use. 'pair' (default) uses data/sample/; "
             "'passage' uses data/sample_passage/.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    split = args.split
    seed = args.seed

    data_dir = _data_dir(split)
    if not (data_dir / "train.jsonl").exists():
        raise FileNotFoundError(
            f"{data_dir}/train.jsonl not found. "
            f"Run 'python -m ml.make_sample_data --split {split}' first."
        )

    # Name suffix mirrors the neural convention: include split only for passage
    suffix = f"_{split}" if split == "passage" else ""
    name_majority   = f"majority{suffix}_s{seed}"
    name_tfidf_lr   = f"tfidf_logreg{suffix}_s{seed}"

    print(f"[baselines] Loading data from {data_dir} ...")
    tr_c, tr_e, tr_y = _load_split(data_dir, "train.jsonl")
    dv_c, dv_e, dv_y = _load_split(data_dir, "dev.jsonl")
    ts_c, ts_e, ts_y = _load_split(data_dir, "test.jsonl")
    print(f"  train={len(tr_y)}  dev={len(dv_y)}  test={len(ts_y)}")

    # --- majority ---
    r_maj = run_majority(
        tr_c, tr_e, tr_y,
        dv_c, dv_e, dv_y,
        ts_c, ts_e, ts_y,
        name=name_majority, seed=seed,
    )
    _write_report(Path("reports") / name_majority, r_maj)
    print(
        f"[{name_majority}] acc={r_maj['accuracy']:.4f}  "
        f"macro-F1={r_maj['macro_f1']:.4f}"
    )

    # --- tfidf_logreg ---
    r_tfidf = run_tfidf_logreg(
        tr_c, tr_e, tr_y,
        dv_c, dv_e, dv_y,
        ts_c, ts_e, ts_y,
        name=name_tfidf_lr, seed=seed,
    )
    _write_report(Path("reports") / name_tfidf_lr, r_tfidf)
    print(
        f"[{name_tfidf_lr}] acc={r_tfidf['accuracy']:.4f}  "
        f"macro-F1={r_tfidf['macro_f1']:.4f}"
    )

    return r_maj, r_tfidf


if __name__ == "__main__":
    main()
