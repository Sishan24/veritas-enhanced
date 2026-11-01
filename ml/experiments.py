"""Run the comparison suites and write reports/comparison.json.

  python -m ml.experiments                    # all suites
  python -m ml.experiments --suites arch      # only architecture comparison
"""
import argparse
import json
from pathlib import Path

from ml.models import ACTIVATIONS, ARCHS
from ml.train import OPTIMIZERS, parse, run

SUITES = {
    "arch": lambda: [(a, ["--arch", a]) for a in ARCHS],
    "optimizer": lambda: [(f"opt_{o}", ["--arch", "bilstm_attn", "--optimizer", o, "--no-save"]) for o in OPTIMIZERS],
    "activation": lambda: [(f"act_{a}", ["--arch", "bilstm_attn", "--activation", a, "--no-save"]) for a in ACTIVATIONS],
    "regularization": lambda: [
        ("reg_none", ["--arch", "bilstm_attn", "--dropout", "0", "--patience", "0", "--no-save"]),
        ("reg_dropout", ["--arch", "bilstm_attn", "--dropout", "0.4", "--patience", "0", "--no-save"]),
        ("reg_wd", ["--arch", "bilstm_attn", "--dropout", "0", "--weight-decay", "1e-4", "--patience", "0", "--no-save"]),
        ("reg_early_stop", ["--arch", "bilstm_attn", "--dropout", "0", "--patience", "3", "--no-save"]),
    ],
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--data-dir", default="data/sample")
    a = ap.parse_args()
    rows = []
    for suite in a.suites:
        for name, extra in SUITES[suite]():
            r = run(parse(["--name", name, "--epochs", str(a.epochs), "--data-dir", a.data_dir, *extra]))
            rows.append({"suite": suite, **{k: v for k, v in r.items() if k not in ("labels",)}})
    Path("reports").mkdir(exist_ok=True)
    Path("reports/comparison.json").write_text(json.dumps(rows, indent=2))
    print("Wrote reports/comparison.json")
