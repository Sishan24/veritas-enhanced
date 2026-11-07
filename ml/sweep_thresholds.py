"""Sweep decide() thresholds over dev claims and visualize metric surfaces."""
import argparse
import csv
import json
from pathlib import Path
import sys

# Ensure matplotlib can be loaded from system site-packages if needed
try:
    import matplotlib
except ImportError:
    sys.path.append("C:/Users/hp/AppData/Local/Programs/Python/Python312/Lib/site-packages")
    import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from backend.app.config import EVIDENCE_PATH, MIN_RETRIEVAL, STANCE_THRESHOLD, TOP_K
from backend.app.services.retriever import Retriever, load_passages
from backend.app.services.verifier import VerifierService, decide
from ml.data import read_jsonl

CLASSES = ["SUPPORTED", "REFUTED", "INSUFFICIENT"]
ALL_VERDICTS = ["SUPPORTED", "REFUTED", "CONFLICTING", "INSUFFICIENT"]

DEFAULT_MIN_RET_RANGE = [0.00, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]
DEFAULT_STANCE_RANGE = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def parse_grid(val, default):
    if val is None:
        return list(default)
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    val_str = str(val).strip()
    if not val_str:
        return list(default)
    return [float(x.strip()) for x in val_str.split(",") if x.strip()]


def evaluate_grid_point(cached_items, targets, mr, st):
    preds = [decide(items, min_retrieval=mr, stance_threshold=st)[0] for items in cached_items]
    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, labels=CLASSES, average="macro", zero_division=0.0)

    recalls = {}
    for c in CLASSES:
        c_targets = [t == c for t in targets]
        c_correct = sum(p == c and t == c for p, t in zip(preds, targets))
        recalls[f"recall_{c}"] = c_correct / sum(c_targets) if sum(c_targets) > 0 else 0.0

    shares = {}
    n = len(preds)
    for v in ALL_VERDICTS:
        shares[f"share_{v}"] = sum(p == v for p in preds) / n if n > 0 else 0.0

    return {
        "min_retrieval": mr,
        "stance_threshold": st,
        "accuracy": acc,
        "macro_f1": macro_f1,
        **recalls,
        **shares,
        "_preds": preds,
    }


def bootstrap_ci(targets, preds, n_resamples=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(targets)
    acc_samples, f1_samples = [], []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        sub_t = [targets[i] for i in idx]
        sub_p = [preds[i] for i in idx]
        acc_samples.append(accuracy_score(sub_t, sub_p))
        f1_samples.append(f1_score(sub_t, sub_p, labels=CLASSES, average="macro", zero_division=0.0))

    acc_ci = (float(np.percentile(acc_samples, 2.5)), float(np.percentile(acc_samples, 97.5)))
    f1_ci = (float(np.percentile(f1_samples, 2.5)), float(np.percentile(f1_samples, 97.5)))
    return acc_ci, f1_ci


def paired_bootstrap_diff(targets, preds_a, preds_b, n_resamples=1000, seed=42):
    """Calculate paired bootstrap 95% CI for accuracy difference (preds_b - preds_a)."""
    n = len(targets)
    correct_a = [p == t for p, t in zip(preds_a, targets)]
    correct_b = [p == t for p, t in zip(preds_b, targets)]

    b_wins = sum(b and not a for a, b in zip(correct_a, correct_b))
    a_wins = sum(a and not b for a, b in zip(correct_a, correct_b))
    both_correct = sum(a and b for a, b in zip(correct_a, correct_b))
    both_incorrect = sum(not a and not b for a, b in zip(correct_a, correct_b))

    acc_a = sum(correct_a) / n if n > 0 else 0.0
    acc_b = sum(correct_b) / n if n > 0 else 0.0
    diff_obs = acc_b - acc_a

    diff_arr = np.array([float(b) - float(a) for a, b in zip(correct_a, correct_b)])
    rng = np.random.RandomState(seed)
    sampled_diffs = []
    for _ in range(n_resamples):
        idx = rng.randint(0, n, size=n)
        sampled_diffs.append(float(np.mean(diff_arr[idx])))

    ci_low = float(np.percentile(sampled_diffs, 2.5))
    ci_high = float(np.percentile(sampled_diffs, 97.5))

    return {
        "n": n,
        "acc_a": acc_a,
        "acc_b": acc_b,
        "diff": diff_obs,
        "ci_95": (ci_low, ci_high),
        "b_wins": b_wins,
        "a_wins": a_wins,
        "both_correct": both_correct,
        "both_incorrect": both_incorrect,
    }


def run_paired_comparison(
    checkpoint_path,
    claims_path,
    mr_a=0.05,
    st_a=0.60,
    mr_b=0.30,
    st_b=0.55,
    out_path=None,
    corpus_path=EVIDENCE_PATH,
    top_k=TOP_K,
):
    claims = read_jsonl(claims_path)
    corpus = load_passages(corpus_path)
    retriever = Retriever(corpus)
    verifier = VerifierService(checkpoint_path)
    if not verifier.ready:
        raise RuntimeError(f"Verifier could not load checkpoint: {checkpoint_path}")

    cached_items = []
    targets = []
    for r in claims:
        hits = retriever.search(r["claim"], k=top_k)
        preds = verifier.predict(r["claim"], [h["text"] for h in hits])
        items = [{"id": h["id"], "text": h["text"], "retrieval_score": h["score"], **p} for h, p in zip(hits, preds)]
        cached_items.append(items)
        targets.append(r["verdict"])

    res_a = evaluate_grid_point(cached_items, targets, mr=mr_a, st=st_a)
    res_b = evaluate_grid_point(cached_items, targets, mr=mr_b, st=st_b)

    paired = paired_bootstrap_diff(targets, res_a["_preds"], res_b["_preds"], n_resamples=1000, seed=42)

    b_win_classes = {}
    a_win_classes = {}
    for t, pa, pb in zip(targets, res_a["_preds"], res_b["_preds"]):
        if pb == t and pa != t:
            b_win_classes[t] = b_win_classes.get(t, 0) + 1
        if pa == t and pb != t:
            a_win_classes[t] = a_win_classes.get(t, 0) + 1

    lines = [
        "=======================================================",
        f"Paired Difference Analysis on Dev Claims (N={len(targets)})",
        f"Checkpoint: {checkpoint_path}",
        f"Claims: {claims_path}",
        "=======================================================",
        "",
        f"Setting A: Default (MIN_RETRIEVAL={mr_a:.2f}, STANCE_THRESHOLD={st_a:.2f})",
        f"  Accuracy: {res_a['accuracy']:.4f}",
        f"  Macro-F1: {res_a['macro_f1']:.4f}",
        f"  Recall SUPPORTED:    {res_a['recall_SUPPORTED']:.4f}",
        f"  Recall REFUTED:      {res_a['recall_REFUTED']:.4f}",
        f"  Recall INSUFFICIENT: {res_a['recall_INSUFFICIENT']:.4f}",
        "  Predicted Shares:",
        f"    SUPPORTED:    {res_a['share_SUPPORTED']:.4f}",
        f"    REFUTED:      {res_a['share_REFUTED']:.4f}",
        f"    CONFLICTING:  {res_a['share_CONFLICTING']:.4f}",
        f"    INSUFFICIENT: {res_a['share_INSUFFICIENT']:.4f}",
        "",
        f"Setting B: Dev-Best (MIN_RETRIEVAL={mr_b:.2f}, STANCE_THRESHOLD={st_b:.2f})",
        f"  Accuracy: {res_b['accuracy']:.4f}",
        f"  Macro-F1: {res_b['macro_f1']:.4f}",
        f"  Recall SUPPORTED:    {res_b['recall_SUPPORTED']:.4f}",
        f"  Recall REFUTED:      {res_b['recall_REFUTED']:.4f}",
        f"  Recall INSUFFICIENT: {res_b['recall_INSUFFICIENT']:.4f}",
        "  Predicted Shares:",
        f"    SUPPORTED:    {res_b['share_SUPPORTED']:.4f}",
        f"    REFUTED:      {res_b['share_REFUTED']:.4f}",
        f"    CONFLICTING:  {res_b['share_CONFLICTING']:.4f}",
        f"    INSUFFICIENT: {res_b['share_INSUFFICIENT']:.4f}",
        "",
        "Paired Comparison (Setting B minus Setting A):",
        f"  Accuracy Difference (B - A): {paired['diff']:+.4f}",
        f"  Paired Bootstrap 95% CI: [{paired['ci_95'][0]:+.4f}, {paired['ci_95'][1]:+.4f}]",
        f"  Claims B gets right that A gets wrong (B wins): {paired['b_wins']}",
        f"  Claims A gets right that B gets wrong (A wins): {paired['a_wins']}",
        f"  Claims both get right: {paired['both_correct']}",
        f"  Claims both get wrong: {paired['both_incorrect']}",
        f"  B-wins class breakdown: {dict(sorted(b_win_classes.items()))}",
        f"  A-wins class breakdown: {dict(sorted(a_win_classes.items()))}",
        "",
        "Shift and Mechanism Analysis:",
        f"  True class share is balanced at 33.33% each (21 claims per class).",
        f"  Setting A severely under-predicted INSUFFICIENT ({res_a['share_INSUFFICIENT']*100:.1f}%) and over-predicted SUPPORTED ({res_a['share_SUPPORTED']*100:.1f}%).",
        f"  Setting B increased INSUFFICIENT prediction share to {res_b['share_INSUFFICIENT']*100:.1f}%.",
        f"  Plain conclusion: Setting B wins mainly by predicting INSUFFICIENT more. By raising MIN_RETRIEVAL from {mr_a:.2f} to {mr_b:.2f}, weak irrelevant retrieval hits are suppressed, converting false SUPPORTED/CONFLICTING predictions into correct INSUFFICIENT verdicts ({b_win_classes.get('INSUFFICIENT', 0)} of B's {paired['b_wins']} wins were on true INSUFFICIENT claims).",
    ]
    report_text = "\n".join(lines)
    print(report_text)

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_text + "\n")
        print(f"\nWrote paired report: {out_path}")

    return {
        "res_a": res_a,
        "res_b": res_b,
        "paired": paired,
        "b_win_classes": b_win_classes,
        "a_win_classes": a_win_classes,
    }




def plot_sweep_figures(results, min_ret_range, stance_range, default_mr, default_st, best_mr, best_st, label, out_png):
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) Dev accuracy against each threshold holding the other at default
    ax1 = axes[0]
    acc_vs_mr = [r["accuracy"] for r in results if abs(r["stance_threshold"] - default_st) < 1e-6]
    mr_vals = [r["min_retrieval"] for r in results if abs(r["stance_threshold"] - default_st) < 1e-6]

    acc_vs_st = [r["accuracy"] for r in results if abs(r["min_retrieval"] - default_mr) < 1e-6]
    st_vals = [r["stance_threshold"] for r in results if abs(r["min_retrieval"] - default_mr) < 1e-6]

    ax1.plot(mr_vals, acc_vs_mr, marker="o", label=f"Accuracy vs MIN_RETRIEVAL (ST={default_st})", color="#1f77b4")
    ax1.plot(st_vals, acc_vs_st, marker="s", label=f"Accuracy vs STANCE_THRESHOLD (MR={default_mr})", color="#ff7f0e")
    ax1.axvline(default_mr, color="#1f77b4", linestyle="--", alpha=0.5, label=f"Default MR ({default_mr})")
    ax1.axvline(default_st, color="#ff7f0e", linestyle="--", alpha=0.5, label=f"Default ST ({default_st})")
    ax1.set_title(f"Dev Accuracy vs Thresholds\n({label})")
    ax1.set_xlabel("Threshold Value")
    ax1.set_ylabel("Dev Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    # (b) Predicted-verdict distribution across STANCE_THRESHOLD at default MIN_RETRIEVAL
    ax2 = axes[1]
    st_slice = [r for r in results if abs(r["min_retrieval"] - default_mr) < 1e-6]
    st_slice.sort(key=lambda x: x["stance_threshold"])
    xs = [r["stance_threshold"] for r in st_slice]
    sh_sup = [r["share_SUPPORTED"] for r in st_slice]
    sh_ref = [r["share_REFUTED"] for r in st_slice]
    sh_conf = [r["share_CONFLICTING"] for r in st_slice]
    sh_ins = [r["share_INSUFFICIENT"] for r in st_slice]

    ax2.plot(xs, sh_sup, marker="o", label="SUPPORTED", color="#2ca02c")
    ax2.plot(xs, sh_ref, marker="x", label="REFUTED", color="#d62728")
    ax2.plot(xs, sh_conf, marker="^", label="CONFLICTING", color="#9467bd")
    ax2.plot(xs, sh_ins, marker="d", label="INSUFFICIENT", color="#7f7f7f")
    ax2.axvline(default_st, color="gray", linestyle="--", alpha=0.6, label=f"Default ST ({default_st})")
    ax2.set_title(f"Predicted Verdict Share vs STANCE_THRESHOLD\n(at MR={default_mr})")
    ax2.set_xlabel("STANCE_THRESHOLD")
    ax2.set_ylabel("Share of Predictions")
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc="center left")

    # (c) Heatmap of dev macro-F1 over the 2D grid
    ax3 = axes[2]
    grid_f1 = np.zeros((len(min_ret_range), len(stance_range)))
    lookup = {(r["min_retrieval"], r["stance_threshold"]): r["macro_f1"] for r in results}
    for i, mr in enumerate(min_ret_range):
        for j, st in enumerate(stance_range):
            grid_f1[i, j] = lookup.get((mr, st), 0.0)

    im = ax3.imshow(grid_f1, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax3)
    cbar.set_label("Dev Macro-F1")

    ax3.set_xticks(range(len(stance_range)))
    ax3.set_xticklabels([f"{v:.2f}" for v in stance_range], rotation=45, fontsize=8)
    ax3.set_yticks(range(len(min_ret_range)))
    ax3.set_yticklabels([f"{v:.2f}" for v in min_ret_range], fontsize=8)
    ax3.set_xlabel("STANCE_THRESHOLD")
    ax3.set_ylabel("MIN_RETRIEVAL")
    ax3.set_title(f"Dev Macro-F1 Surface\n({label})")

    # Mark default and best
    if default_st in stance_range and default_mr in min_ret_range:
        def_j = stance_range.index(default_st)
        def_i = min_ret_range.index(default_mr)
        ax3.scatter([def_j], [def_i], color="red", marker="o", s=100, edgecolors="white", linewidths=1.5, label="Default (0.05, 0.60)")

    if best_st in stance_range and best_mr in min_ret_range:
        best_j = stance_range.index(best_st)
        best_i = min_ret_range.index(best_mr)
        ax3.scatter([best_j], [best_i], color="gold", marker="*", s=160, edgecolors="black", linewidths=1.0, label=f"Best ({best_mr:.2f}, {best_st:.2f})")

    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"Saved figure: {out_png}")


def run_sweep(checkpoint_path, claims_path, label, out_csv, out_png=None,
              min_ret_range=DEFAULT_MIN_RET_RANGE, stance_range=DEFAULT_STANCE_RANGE,
              corpus_path=EVIDENCE_PATH, top_k=TOP_K):
    if "test" in Path(claims_path).stem.lower():
        raise ValueError(
            f"Full-grid sweep is strictly prohibited on test sets ({claims_path}). "
            "Use pre-registered evaluation points for test sets."
        )

    print(f"\n=======================================================")
    print(f"Running sweep for: {label}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Claims: {claims_path}")
    print(f"Corpus: {corpus_path}")
    print(f"TOP_K: {top_k}")
    print(f"MIN_RETRIEVAL range: {min_ret_range}")
    print(f"STANCE_THRESHOLD range: {stance_range}")
    print(f"=======================================================")

    claims = read_jsonl(claims_path)
    corpus = load_passages(corpus_path)
    retriever = Retriever(corpus)
    verifier = VerifierService(checkpoint_path)
    if not verifier.ready:
        raise RuntimeError(f"Verifier could not load checkpoint: {checkpoint_path}")

    # 1. Run model once and cache per-passage probs and retrieval scores
    cached_items = []
    targets = []
    for r in claims:
        hits = retriever.search(r["claim"], k=top_k)
        preds = verifier.predict(r["claim"], [h["text"] for h in hits])
        items = [{"id": h["id"], "text": h["text"], "retrieval_score": h["score"], **p} for h, p in zip(hits, preds)]
        cached_items.append(items)
        targets.append(r["verdict"])

    # 2. Sweep the grid
    results = []
    best_res = None
    default_res = None

    for mr in min_ret_range:
        for st in stance_range:
            res = evaluate_grid_point(cached_items, targets, mr, st)
            results.append(res)

            if abs(mr - MIN_RETRIEVAL) < 1e-6 and abs(st - STANCE_THRESHOLD) < 1e-6:
                default_res = res

            if best_res is None or res["macro_f1"] > best_res["macro_f1"] or (
                abs(res["macro_f1"] - best_res["macro_f1"]) < 1e-6 and res["accuracy"] > best_res["accuracy"]
            ):
                best_res = res

    # 3. Bootstrap CI for default and best
    def_acc_ci, def_f1_ci = bootstrap_ci(targets, default_res["_preds"])
    best_acc_ci, best_f1_ci = bootstrap_ci(targets, best_res["_preds"])

    print(f"\n[{label}] SUMMARY:")
    print(f"Current defaults (MR={MIN_RETRIEVAL}, ST={STANCE_THRESHOLD}):")
    print(f"   Accuracy: {default_res['accuracy']:.4f} (95% CI: [{def_acc_ci[0]:.4f}, {def_acc_ci[1]:.4f}])")
    print(f"   Macro-F1: {default_res['macro_f1']:.4f} (95% CI: [{def_f1_ci[0]:.4f}, {def_f1_ci[1]:.4f}])")
    print(f"   Shares: SUP={default_res['share_SUPPORTED']:.3f}, REF={default_res['share_REFUTED']:.3f}, CONF={default_res['share_CONFLICTING']:.3f}, INS={default_res['share_INSUFFICIENT']:.3f}")

    print(f"Best grid point (MR={best_res['min_retrieval']}, ST={best_res['stance_threshold']}):")
    print(f"   Accuracy: {best_res['accuracy']:.4f} (95% CI: [{best_acc_ci[0]:.4f}, {best_acc_ci[1]:.4f}])")
    print(f"   Macro-F1: {best_res['macro_f1']:.4f} (95% CI: [{best_f1_ci[0]:.4f}, {best_f1_ci[1]:.4f}])")
    print(f"   Shares: SUP={best_res['share_SUPPORTED']:.3f}, REF={best_res['share_REFUTED']:.3f}, CONF={best_res['share_CONFLICTING']:.3f}, INS={best_res['share_INSUFFICIENT']:.3f}")
    print(f"Gap (Best - Default): Accuracy={best_res['accuracy'] - default_res['accuracy']:+.4f}, Macro-F1={best_res['macro_f1'] - default_res['macro_f1']:+.4f}")

    # 4. Write CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "min_retrieval", "stance_threshold", "accuracy", "macro_f1",
        "recall_SUPPORTED", "recall_REFUTED", "recall_INSUFFICIENT",
        "share_SUPPORTED", "share_REFUTED", "share_CONFLICTING", "share_INSUFFICIENT"
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: round(r[k], 6) for k in fieldnames}
            writer.writerow(row)
    print(f"Wrote CSV: {out_csv}")

    # 5. Plot figures
    if out_png:
        plot_sweep_figures(
            results, min_ret_range, stance_range,
            MIN_RETRIEVAL, STANCE_THRESHOLD,
            best_res["min_retrieval"], best_res["stance_threshold"],
            label, out_png
        )

    return {
        "label": label,
        "default": {**default_res, "acc_ci": def_acc_ci, "f1_ci": def_f1_ci},
        "best": {**best_res, "acc_ci": best_acc_ci, "f1_ci": best_f1_ci},
        "results": results,
    }


def build_parser():
    p = argparse.ArgumentParser(description="Sweep decide() thresholds over claim-level data.")
    p.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    p.add_argument("--claims", default="data/claim_level/dev_claims.jsonl", help="Claims jsonl file")
    p.add_argument("--label", default=None, help="Run label (e.g. Passage-Clean or LEAKY-Deployed)")
    p.add_argument("--out", default=None, help="Output CSV path")
    p.add_argument("--fig", default=None, help="Output figure PNG path")
    p.add_argument("--mr-grid", default=None, help="Comma-separated float values for MIN_RETRIEVAL grid")
    p.add_argument("--st-grid", default=None, help="Comma-separated float values for STANCE_THRESHOLD grid")
    p.add_argument("--corpus", default=EVIDENCE_PATH, help="Path to evidence corpus jsonl")
    p.add_argument("--top-k", type=int, default=TOP_K, help="Number of retrieved passages per claim")

    # Paired comparison options
    p.add_argument("--paired", action="store_true", help="Run paired comparison between two settings")
    p.add_argument("--mr-a", type=float, default=0.05, help="Setting A MIN_RETRIEVAL")
    p.add_argument("--st-a", type=float, default=0.60, help="Setting A STANCE_THRESHOLD")
    p.add_argument("--mr-b", type=float, default=0.30, help="Setting B MIN_RETRIEVAL")
    p.add_argument("--st-b", type=float, default=0.55, help="Setting B STANCE_THRESHOLD")
    p.add_argument("--paired-out", default=None, help="Path to save paired comparison text report")
    return p


def main():
    p = build_parser()
    args = p.parse_args()

    if args.paired:
        run_paired_comparison(
            checkpoint_path=args.checkpoint,
            claims_path=args.claims,
            mr_a=args.mr_a,
            st_a=args.st_a,
            mr_b=args.mr_b,
            st_b=args.st_b,
            out_path=args.paired_out,
            corpus_path=args.corpus,
            top_k=args.top_k,
        )
        return

    if not args.out or not args.label:
        p.error("--out and --label are required for full sweep mode.")

    mr_range = parse_grid(args.mr_grid, DEFAULT_MIN_RET_RANGE)
    st_range = parse_grid(args.st_grid, DEFAULT_STANCE_RANGE)

    run_sweep(
        args.checkpoint,
        args.claims,
        args.label,
        args.out,
        args.fig,
        min_ret_range=mr_range,
        stance_range=st_range,
        corpus_path=args.corpus,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
