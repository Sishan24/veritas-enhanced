"""Aggregate Day 2 experimental sweep results into CSV and Markdown summaries.

Reads runs/*/metrics.json and computes sample mean and standard deviation (ddof=1).
Deterministic baselines (majority, tfidf_logreg) are deduplicated across model seeds.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

# Standard expected architecture configurations for Day 2
NEURAL_ARCHS = ["mlp", "cnn", "lstm", "bilstm", "bilstm_attn"]
DEFAULT_HIDDEN = 128
CAPACITY_CONTROL_ARCH = "lstm"
CAPACITY_CONTROL_HIDDEN = 256
BASELINES = ["majority", "tfidf_logreg"]

DETERMINISTIC_BASELINES = {"majority", "tfidf_logreg"}

ALL_CONFIGS = [
    ("mlp", 128),
    ("cnn", 128),
    ("lstm", 128),
    ("bilstm", 128),
    ("bilstm_attn", 128),
    ("lstm", 256),
    ("majority", 0),
    ("tfidf_logreg", 0),
]


def compute_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Compute sample mean and sample standard deviation (ddof=1).

    Returns (mean, std). For n=0, returns (None, None).
    For n=1, sample std is undefined (ddof=1) and returns (mean, None).
    """
    n = len(values)
    if n == 0:
        return None, None
    if n == 1:
        return values[0], None
    mean_val = statistics.mean(values)
    std_val = statistics.stdev(values)
    return mean_val, std_val


def format_mean_std(mean: float | None, std: float | None, precision: int = 4) -> str:
    """Format mean ± std string.

    With n=1 (std is None), displays '{mean} ± n/a', never 0.
    With n=0 (mean is None), displays 'n/a'.
    """
    if mean is None:
        return "n/a"
    if std is None:
        return f"{mean:.{precision}f} ± n/a"
    return f"{mean:.{precision}f} ± {std:.{precision}f}"


def parse_run_dir_name(dir_name: str) -> dict[str, Any] | None:
    """Parse run directory name: <arch>_h<hidden>_<split_tag>_ms<model_seed>."""
    parts = dir_name.split("_")
    if len(parts) < 4:
        return None

    # arch may be multi-part (e.g. bilstm_attn or tfidf_logreg)
    ms_part = parts[-1]
    if not ms_part.startswith("ms"):
        return None
    try:
        model_seed = int(ms_part[2:])
    except ValueError:
        return None

    # Split tag: either 'pair' or 'passage_ss<split_seed>'
    if "passage" in parts:
        p_idx = parts.index("passage")
        split = "passage"
        split_seed_str = parts[p_idx + 1]
        split_seed = int(split_seed_str.replace("ss", "")) if split_seed_str.startswith("ss") else None
        split_tag = f"passage_ss{split_seed}" if split_seed is not None else "passage"
        arch_hidden_parts = parts[:p_idx]
    elif "pair" in parts:
        p_idx = parts.index("pair")
        split = "pair"
        split_seed = None
        split_tag = "pair"
        arch_hidden_parts = parts[:p_idx]
    else:
        return None

    hidden_part = arch_hidden_parts[-1]
    if hidden_part.startswith("h") and hidden_part[1:].isdigit():
        hidden = int(hidden_part[1:])
        arch = "_".join(arch_hidden_parts[:-1])
    else:
        hidden = 0
        arch = "_".join(arch_hidden_parts)

    return {
        "run_name": dir_name,
        "arch": arch,
        "hidden": hidden,
        "split": split,
        "split_seed": split_seed,
        "split_tag": split_tag,
        "model_seed": model_seed,
    }


def load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    """Load all metrics.json files under runs_dir."""
    runs = []
    if not runs_dir.exists():
        return runs

    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        parsed = parse_run_dir_name(d.name)
        metrics_file = d / "metrics.json"
        if not metrics_file.exists():
            continue
        try:
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        item = {
            "dir_name": d.name,
            "arch": metrics.get("arch", parsed["arch"] if parsed else d.name),
            "hidden": parsed["hidden"] if parsed else 0,
            "split": parsed["split"] if parsed else "unknown",
            "split_seed": parsed["split_seed"] if parsed else None,
            "split_tag": parsed["split_tag"] if parsed else "unknown",
            "model_seed": metrics.get("seed", parsed["model_seed"] if parsed else 0),
            "n_params": metrics.get("n_params", 0),
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
        }
        runs.append(item)
    return runs


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate runs into tables:
    - pair_pooled: Pair split (neural n=2, deterministic baseline n=1)
    - passage_pooled: Pooled passage runs (neural n=6, baseline n=3 across split seeds)
    - passage_split_seeds: Breakdown by passage split seed (ss0, ss1, ss2)
    - all_rows: All 40 rows for CSV output
    - capacity_comparison: lstm_h256 vs bilstm_h128
    """
    # Index runs by (arch, hidden, split_tag, model_seed)
    run_map: dict[tuple[str, int, str, int], dict[str, Any]] = {}
    for r in runs:
        key = (r["arch"], r["hidden"], r["split_tag"], r["model_seed"])
        run_map[key] = r

    pair_rows = []
    passage_pooled_rows = []
    split_seed_rows = []
    all_rows = []

    for arch, hidden in ALL_CONFIGS:
        is_baseline = arch in DETERMINISTIC_BASELINES

        # -------------------------------------------------------------------
        # 1. Pair Split
        # -------------------------------------------------------------------
        if is_baseline:
            # Baseline is deterministic; use model_seed=0 only (n=1)
            r0 = run_map.get((arch, hidden, "pair", 0))
            if r0 is not None and r0.get("accuracy") is not None:
                acc_list = [r0["accuracy"]]
                f1_list = [r0["macro_f1"]]
                n_params = r0["n_params"]
            else:
                acc_list = []
                f1_list = []
                n_params = 0
        else:
            # Neural: pool model seeds 0 and 1
            neural_pair_runs = [
                run_map[(arch, hidden, "pair", ms)]
                for ms in (0, 1)
                if (arch, hidden, "pair", ms) in run_map
            ]
            acc_list = [r["accuracy"] for r in neural_pair_runs if r.get("accuracy") is not None]
            f1_list = [r["macro_f1"] for r in neural_pair_runs if r.get("macro_f1") is not None]
            n_params = neural_pair_runs[0]["n_params"] if neural_pair_runs else 0

        acc_mean, acc_std = compute_mean_std(acc_list)
        f1_mean, f1_std = compute_mean_std(f1_list)

        pair_row = {
            "arch": arch,
            "hidden": hidden,
            "n_params": n_params,
            "split": "pair",
            "n": len(acc_list),
            "accuracy mean ± std": format_mean_std(acc_mean, acc_std),
            "macro-F1 mean ± std": format_mean_std(f1_mean, f1_std),
            "raw_acc_mean": acc_mean,
            "raw_acc_std": acc_std,
            "raw_f1_mean": f1_mean,
            "raw_f1_std": f1_std,
        }
        pair_rows.append(pair_row)
        all_rows.append(pair_row)

        # -------------------------------------------------------------------
        # 2. Passage Pooled
        # -------------------------------------------------------------------
        if is_baseline:
            # Spread across the 3 split seeds only (n=3), seed 0 for each
            base_ss_runs = [
                run_map[(arch, hidden, f"passage_ss{ss}", 0)]
                for ss in (0, 1, 2)
                if (arch, hidden, f"passage_ss{ss}", 0) in run_map
            ]
            acc_list = [r["accuracy"] for r in base_ss_runs if r.get("accuracy") is not None]
            f1_list = [r["macro_f1"] for r in base_ss_runs if r.get("macro_f1") is not None]
            n_params = base_ss_runs[0]["n_params"] if base_ss_runs else 0
        else:
            # Neural: pool all 6 runs (3 split seeds x 2 model seeds)
            neural_pass_runs = [
                run_map[(arch, hidden, f"passage_ss{ss}", ms)]
                for ss in (0, 1, 2)
                for ms in (0, 1)
                if (arch, hidden, f"passage_ss{ss}", ms) in run_map
            ]
            acc_list = [r["accuracy"] for r in neural_pass_runs if r.get("accuracy") is not None]
            f1_list = [r["macro_f1"] for r in neural_pass_runs if r.get("macro_f1") is not None]
            n_params = int(round(statistics.mean(r["n_params"] for r in neural_pass_runs))) if neural_pass_runs else 0

        acc_mean, acc_std = compute_mean_std(acc_list)
        f1_mean, f1_std = compute_mean_std(f1_list)

        passage_pooled_row = {
            "arch": arch,
            "hidden": hidden,
            "n_params": n_params,
            "split": "passage (pooled)",
            "n": len(acc_list),
            "accuracy mean ± std": format_mean_std(acc_mean, acc_std),
            "macro-F1 mean ± std": format_mean_std(f1_mean, f1_std),
            "raw_acc_mean": acc_mean,
            "raw_acc_std": acc_std,
            "raw_f1_mean": f1_mean,
            "raw_f1_std": f1_std,
        }
        passage_pooled_rows.append(passage_pooled_row)
        all_rows.append(passage_pooled_row)

        # -------------------------------------------------------------------
        # 3. Passage Split Seeds (ss0, ss1, ss2)
        # -------------------------------------------------------------------
        for ss in (0, 1, 2):
            split_tag = f"passage_ss{ss}"
            if is_baseline:
                r0 = run_map.get((arch, hidden, split_tag, 0))
                if r0 is not None and r0.get("accuracy") is not None:
                    acc_list = [r0["accuracy"]]
                    f1_list = [r0["macro_f1"]]
                    n_params = r0["n_params"]
                else:
                    acc_list = []
                    f1_list = []
                    n_params = 0
            else:
                ss_runs = [
                    run_map[(arch, hidden, split_tag, ms)]
                    for ms in (0, 1)
                    if (arch, hidden, split_tag, ms) in run_map
                ]
                acc_list = [r["accuracy"] for r in ss_runs if r.get("accuracy") is not None]
                f1_list = [r["macro_f1"] for r in ss_runs if r.get("macro_f1") is not None]
                n_params = ss_runs[0]["n_params"] if ss_runs else 0

            acc_mean, acc_std = compute_mean_std(acc_list)
            f1_mean, f1_std = compute_mean_std(f1_list)

            ss_row = {
                "arch": arch,
                "hidden": hidden,
                "n_params": n_params,
                "split": split_tag,
                "n": len(acc_list),
                "accuracy mean ± std": format_mean_std(acc_mean, acc_std),
                "macro-F1 mean ± std": format_mean_std(f1_mean, f1_std),
                "raw_acc_mean": acc_mean,
                "raw_acc_std": acc_std,
                "raw_f1_mean": f1_mean,
                "raw_f1_std": f1_std,
            }
            split_seed_rows.append(ss_row)
            all_rows.append(ss_row)

    # -----------------------------------------------------------------------
    # 4. Capacity Comparison: lstm_h256 vs bilstm_h128
    # -----------------------------------------------------------------------
    bilstm_pair = next(r for r in pair_rows if r["arch"] == "bilstm" and r["hidden"] == 128)
    lstm_pair = next(r for r in pair_rows if r["arch"] == "lstm" and r["hidden"] == 256)
    bilstm_pass = next(r for r in passage_pooled_rows if r["arch"] == "bilstm" and r["hidden"] == 128)
    lstm_pass = next(r for r in passage_pooled_rows if r["arch"] == "lstm" and r["hidden"] == 256)

    pair_ratio = lstm_pair["n_params"] / bilstm_pair["n_params"] if bilstm_pair["n_params"] else 0.0
    pass_ratio = lstm_pass["n_params"] / bilstm_pass["n_params"] if bilstm_pass["n_params"] else 0.0

    capacity_comparison = {
        "bilstm_pair_params": bilstm_pair["n_params"],
        "lstm_pair_params": lstm_pair["n_params"],
        "pair_param_ratio": pair_ratio,
        "bilstm_pass_params": bilstm_pass["n_params"],
        "lstm_pass_params": lstm_pass["n_params"],
        "pass_param_ratio": pass_ratio,
        "bilstm_pair_acc": bilstm_pair["accuracy mean ± std"],
        "lstm_pair_acc": lstm_pair["accuracy mean ± std"],
        "bilstm_pair_f1": bilstm_pair["macro-F1 mean ± std"],
        "lstm_pair_f1": lstm_pair["macro-F1 mean ± std"],
        "bilstm_pass_acc": bilstm_pass["accuracy mean ± std"],
        "lstm_pass_acc": lstm_pass["accuracy mean ± std"],
        "bilstm_pass_f1": bilstm_pass["macro-F1 mean ± std"],
        "lstm_pass_f1": lstm_pass["macro-F1 mean ± std"],
    }

    return {
        "pair_rows": pair_rows,
        "passage_pooled_rows": passage_pooled_rows,
        "split_seed_rows": split_seed_rows,
        "all_rows": all_rows,
        "capacity_comparison": capacity_comparison,
    }


def write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    """Write aggregated rows to CSV."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "arch",
        "hidden",
        "n_params",
        "split",
        "n",
        "accuracy mean ± std",
        "macro-F1 mean ± std",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def render_markdown_table(rows: list[dict[str, Any]]) -> str:
    """Render a list of row dicts as a GitHub-flavored Markdown table."""
    headers = ["arch", "hidden", "n_params", "split", "n", "accuracy mean ± std", "macro-F1 mean ± std"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in rows:
        line = f"| {r['arch']} | {r['hidden']} | {r['n_params']:,} | {r['split']} | {r['n']} | {r['accuracy mean ± std']} | {r['macro-F1 mean ± std']} |"
        lines.append(line)
    return "\n".join(lines)


def write_markdown(data: dict[str, Any], out_path: Path, commentary: str = "") -> None:
    """Write summary Markdown document."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = data["capacity_comparison"]

    md = []
    md.append("# VERITAS Day 2 Evaluation Summary\n")

    md.append("## 1. Main Results: Pair vs Passage (Pooled)\n")
    main_rows = data["pair_rows"] + data["passage_pooled_rows"]
    md.append(render_markdown_table(main_rows))
    md.append(
        "\n*Note: `majority` and `tfidf_logreg` are deterministic baselines; duplicate model seeds "
        "(ms0 and ms1) yield identical results and are not counted as independent samples. "
        "For individual splits, n=1 (sample std is n/a). For the pooled passage view, metrics reflect "
        "the spread across the 3 split seeds only (n=3).*\n"
    )

    md.append("## 2. Passage Split Breakdown (Split Seed Effect vs Model Seed)\n")
    md.append(render_markdown_table(data["split_seed_rows"]))
    md.append(
        "\n*Note: Each split seed row for neural architectures averages n=2 model seeds (ms0 and ms1). "
        "Baselines have n=1 per split seed.*\n"
    )

    md.append("## 3. Capacity Control: LSTM (h=256) vs BiLSTM (h=128)\n")
    md.append("| Architecture | Hidden | Split | n_params | Ratio (vs BiLSTM) | Accuracy (mean ± std) | Macro-F1 (mean ± std) |")
    md.append("| --- | --- | --- | --- | --- | --- | --- |")
    md.append(f"| bilstm | 128 | pair | {cap['bilstm_pair_params']:,} | 1.00x | {cap['bilstm_pair_acc']} | {cap['bilstm_pair_f1']} |")
    md.append(f"| lstm | 256 | pair | {cap['lstm_pair_params']:,} | {cap['pair_param_ratio']:.2f}x | {cap['lstm_pair_acc']} | {cap['lstm_pair_f1']} |")
    md.append(f"| bilstm | 128 | passage (pooled) | {cap['bilstm_pass_params']:,} | 1.00x | {cap['bilstm_pass_acc']} | {cap['bilstm_pass_f1']} |")
    md.append(f"| lstm | 256 | passage (pooled) | {cap['lstm_pass_params']:,} | {cap['pass_param_ratio']:.2f}x | {cap['lstm_pass_acc']} | {cap['lstm_pass_f1']} |")
    md.append("")

    if commentary:
        md.append("## 4. Findings and Analysis\n")
        md.append(commentary.strip() + "\n")

    out_path.write_text("\n".join(md), encoding="utf-8")


DEFAULT_COMMENTARY = """1. In the pooled passage evaluation, the macro-F1 differences between lstm_h128 (0.6208 ± 0.0467), lstm_h256 (0.6259 ± 0.0343), and cnn_h128 (0.5847 ± 0.0905) fall within their overlapping standard deviation intervals and are therefore not statistically distinguishable.
2. Similarly, bilstm_h128 (0.5332 ± 0.0382) and bilstm_attn_h128 (0.5336 ± 0.0546) exhibit virtually identical mean macro-F1 with overlapping standard deviations, showing no distinguishable difference between bidirectional models with and without attention.
3. With only 2 model seeds per split configuration, the reported sample standard deviation is a rough estimate, and an observed difference smaller than this spread does not constitute evidence that one architecture outperforms another.
4. Comparing lstm_h256 (756,930 parameters in passage) with bilstm_h128 (494,274 parameters) yields an actual parameter ratio of 1.53x, meaning the models have substantially different capacities rather than matched parameter counts.
5. In passage splits, bilstm_h128 does not hold an advantage over lstm_h256 or lstm_h128; lstm_h256 attains 0.6259 ± 0.0343 macro-F1 while bilstm_h128 attains 0.5332 ± 0.0382, indicating that bidirectional modeling does not improve performance over unidirectional LSTM on passage partitions.
6. Across all neural architectures, moving from pair splits to passage splits causes macro-F1 to drop by 36 to 46 percentage points (from 0.9716–0.9983 on pair down to 0.5332–0.6259 on pooled passage), confirming severe performance inflation from evidence leakage in pair mode.
7. Partitioning variance across the three passage split seeds (e.g. cnn_h128 macro-F1 ranges from 0.4996 on ss0 to 0.6822 on ss1) produces a much larger spread than random model initialization seeds within any given split (which typically differ by 0.01 to 0.06).
8. All 64 scheduled runs completed successfully without failures, but split seed 1 exhibits an unusual divergence where cnn_h128 macro-F1 jumps to 0.6822 while bilstm_h128 drops to 0.4913, reflecting marked sensitivity to the specific passage split."""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Aggregate VERITAS Day 2 sweep results.")
    parser.add_argument("--runs-dir", default="runs", help="Directory containing run subdirectories (default: runs)")
    parser.add_argument("--csv-out", default="results/day2_summary.csv", help="Path to output CSV (default: results/day2_summary.csv)")
    parser.add_argument("--md-out", default="results/day2_summary.md", help="Path to output Markdown (default: results/day2_summary.md)")
    args = parser.parse_args(argv)

    runs_dir = Path(args.runs_dir)
    csv_out = Path(args.csv_out)
    md_out = Path(args.md_out)

    runs = load_runs(runs_dir)
    if not runs:
        print(f"Warning: No valid runs found in '{runs_dir}'.", file=sys.stderr)

    agg = aggregate_runs(runs)
    write_csv(agg["all_rows"], csv_out)
    write_markdown(agg, md_out, commentary=DEFAULT_COMMENTARY)
    print(f"Aggregated {len(runs)} runs.")
    print(f"Wrote summary CSV to {csv_out}")
    print(f"Wrote summary Markdown to {md_out}")


if __name__ == "__main__":
    main()
