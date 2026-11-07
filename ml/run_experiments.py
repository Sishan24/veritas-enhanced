"""Experiment runner for VERITAS Day 2 evaluation.

Runs neural architectures and baselines across multiple random seeds
and multiple passage-level splits, logging every run to results/run_log.csv.

Usage:
  python -m ml.run_experiments --dry-run
  python -m ml.run_experiments --only-run bilstm_attn_h128_passage_ss0_ms0
  python -m ml.run_experiments
"""
import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

NEURAL_ARCHS = ["mlp", "cnn", "lstm", "bilstm", "bilstm_attn"]
DEFAULT_HIDDEN = 128
CAPACITY_CONTROL_ARCH = "lstm"
CAPACITY_CONTROL_HIDDEN = 256

BASELINES = ["majority", "tfidf_logreg"]

MODEL_SEEDS = [0, 1, 2]

SPLITS = [
    {"split": "pair", "split_seed": None, "data_dir": "data/sample"},
    {"split": "passage", "split_seed": 0, "data_dir": "data/sample_passage"},
    {"split": "passage", "split_seed": 1, "data_dir": "data/sample_passage_ss1"},
    {"split": "passage", "split_seed": 2, "data_dir": "data/sample_passage_ss2"},
]


def make_run_name(arch: str, hidden: int, split: str, split_seed: int | None, model_seed: int) -> str:
    """Generate a unique, explicit run name."""
    if split == "passage":
        split_tag = f"passage_ss{split_seed}"
    else:
        split_tag = "pair"
    return f"{arch}_h{hidden}_{split_tag}_ms{model_seed}"


def get_all_configs(model_seeds: list[int] | None = None) -> list[dict]:
    """Generate the full list of experimental configurations."""
    if model_seeds is None:
        model_seeds = MODEL_SEEDS
    configs = []

    # 1. Neural models (5 default + 1 capacity control = 6 configs)
    neural_configs = [(arch, DEFAULT_HIDDEN) for arch in NEURAL_ARCHS]
    neural_configs.append((CAPACITY_CONTROL_ARCH, CAPACITY_CONTROL_HIDDEN))

    for arch, hidden in neural_configs:
        for split_info in SPLITS:
            for ms in model_seeds:
                name = make_run_name(arch, hidden, split_info["split"], split_info["split_seed"], ms)
                configs.append({
                    "name": name,
                    "kind": "neural",
                    "arch": arch,
                    "hidden": hidden,
                    "split": split_info["split"],
                    "split_seed": split_info["split_seed"],
                    "data_dir": split_info["data_dir"],
                    "model_seed": ms,
                })

    # 2. Baselines (2 baselines)
    for base in BASELINES:
        for split_info in SPLITS:
            for ms in model_seeds:
                name = make_run_name(base, 0, split_info["split"], split_info["split_seed"], ms)
                configs.append({
                    "name": name,
                    "kind": "baseline",
                    "arch": base,
                    "hidden": 0,
                    "split": split_info["split"],
                    "split_seed": split_info["split_seed"],
                    "data_dir": split_info["data_dir"],
                    "model_seed": ms,
                })

    return configs


def build_command(cfg: dict, out_dir: str) -> list[str]:
    """Build subprocess command for a run configuration."""
    if cfg["kind"] == "neural":
        return [
            sys.executable, "-m", "ml.train",
            "--arch", cfg["arch"],
            "--hidden", str(cfg["hidden"]),
            "--seed", str(cfg["model_seed"]),
            "--name", cfg["name"],
            "--out-dir", out_dir,
            "--data-dir", cfg["data_dir"],
            "--split", cfg["split"],
            "--no-save",
        ]
    else:
        return [
            sys.executable, "-m", "ml.baselines",
            "--baseline", cfg["arch"],
            "--seed", str(cfg["model_seed"]),
            "--name", cfg["name"],
            "--out-dir", out_dir,
            "--data-dir", cfg["data_dir"],
            "--split", cfg["split"],
        ]


def append_log(log_path: Path, run_name: str, status: str, seconds: float, error: str = "") -> None:
    """Append a record to results/run_log.csv."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["run_name", "status", "seconds", "error"])
        writer.writerow([run_name, status, f"{seconds:.2f}", error.strip().replace("\n", " ")])


def execute_run(cfg: dict, out_dir: str, log_path: Path, runner_fn=subprocess.run) -> tuple[str, float, str]:
    """Execute a single run, handling skip/overwrite rules and retries.

    Returns (status, elapsed_seconds, error_msg).
    """
    run_dir = Path(out_dir) / cfg["name"]
    metrics_file = run_dir / "metrics.json"

    # Rule: SKIP completed runs
    if metrics_file.exists():
        print(f"[{cfg['name']}] SKIPPED: completed run exists at {run_dir}")
        append_log(log_path, cfg["name"], "SKIPPED", 0.0, "Already completed")
        return "SKIPPED", 0.0, ""

    # Rule: REFUSE to overwrite existing incomplete directory
    if run_dir.exists() and any(run_dir.iterdir()):
        msg = f"Refusing to overwrite existing non-empty directory {run_dir}"
        print(f"[{cfg['name']}] ERROR: {msg}")
        append_log(log_path, cfg["name"], "ERROR_REFUSED_OVERWRITE", 0.0, msg)
        return "ERROR_REFUSED_OVERWRITE", 0.0, msg

    cmd = build_command(cfg, out_dir)
    print(f"[{cfg['name']}] Starting command: {' '.join(cmd)}")

    attempts = 0
    max_attempts = 2  # retry at most once
    status = "FAILED"
    err_text = ""
    elapsed = 0.0

    while attempts < max_attempts:
        attempts += 1
        t0 = time.perf_counter()
        proc = runner_fn(cmd, capture_output=True, text=True)
        elapsed = time.perf_counter() - t0

        if proc.returncode == 0 and metrics_file.exists():
            status = "SUCCESS"
            err_text = ""
            print(f"[{cfg['name']}] SUCCESS in {elapsed:.2f}s")
            break
        else:
            err_text = (proc.stderr or proc.stdout or f"Process exited with {proc.returncode}").strip()
            print(f"[{cfg['name']}] Attempt {attempts} failed in {elapsed:.2f}s: {err_text[:200]}")
            if attempts < max_attempts:
                print(f"[{cfg['name']}] Retrying once...")

    append_log(log_path, cfg["name"], status, elapsed, err_text)
    return status, elapsed, err_text


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run VERITAS Day 2 experiments.")
    p.add_argument("--dry-run", action="store_true", help="Print experiment plan and count without running")
    p.add_argument("--out-dir", default="runs", help="Output directory for runs (default: runs)")
    p.add_argument("--log-file", default="results/run_log.csv", help="Path to run log (default: results/run_log.csv)")
    p.add_argument("--only-run", default=None, help="Execute only the specified run name")
    p.add_argument(
        "--model-seeds",
        nargs="+",
        type=int,
        default=MODEL_SEEDS,
        help="Model seeds to run (default: 0 1 2)",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configs = get_all_configs(model_seeds=args.model_seeds)

    if args.only_run:
        matched = [c for c in configs if c["name"] == args.only_run]
        if not matched:
            print(f"Error: run name '{args.only_run}' not found in experiment configs.", file=sys.stderr)
            sys.exit(1)
        configs = matched

    if args.dry_run:
        print(f"Total planned runs: {len(configs)}")
        print(f"Output directory: {args.out_dir}")
        print(f"Log file: {args.log_file}\n")
        print(f"{'#':<4} {'Run Name':<40} {'Kind':<10} {'Arch':<12} {'Hidden':<6} {'Split':<10} {'Seed':<4}")
        print("-" * 90)
        for i, c in enumerate(configs, 1):
            s_name = c["split"] if c["split_seed"] is None else f"passage_ss{c['split_seed']}"
            print(f"{i:<4} {c['name']:<40} {c['kind']:<10} {c['arch']:<12} {c['hidden']:<6} {s_name:<10} {c['model_seed']:<4}")
        return configs

    out_dir = args.out_dir
    log_path = Path(args.log_file)
    print(f"Running {len(configs)} experiments into '{out_dir}'...")

    total_start = time.perf_counter()
    success_count = 0
    skip_count = 0
    fail_count = 0

    for idx, cfg in enumerate(configs, 1):
        print(f"\n--- [{idx}/{len(configs)}] {cfg['name']} ---")
        status, secs, err = execute_run(cfg, out_dir, log_path)
        if status == "SUCCESS":
            success_count += 1
        elif status == "SKIPPED":
            skip_count += 1
        else:
            fail_count += 1

    total_time = time.perf_counter() - total_start
    print(f"\nSweep completed in {total_time:.2f}s.")
    print(f"Results: {success_count} succeeded, {skip_count} skipped, {fail_count} failed.")
    return configs


if __name__ == "__main__":
    main()
