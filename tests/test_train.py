"""Tests for ml/train.py seeding and reproducibility."""
import json
import shutil
from pathlib import Path
import pytest
from ml.train import parse, run

def test_identical_seed_produces_identical_metrics(tmp_path):
    rows = [
        {"claim": "Paris is in France", "evidence": "Paris is the capital of France.", "label": "SUPPORTED"},
        {"claim": "Berlin is in France", "evidence": "Berlin is the capital of Germany.", "label": "REFUTED"},
        {"claim": "Rome is ancient", "evidence": "Apples grow on trees.", "label": "NOT_ENOUGH_INFO"},
        {"claim": "Tokyo is in Japan", "evidence": "Tokyo is the capital of Japan.", "label": "SUPPORTED"},
    ]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for fname in ("train.jsonl", "dev.jsonl", "test.jsonl"):
        (data_dir / fname).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # Run 1
    args1 = parse([
        "--arch", "mlp",
        "--epochs", "2",
        "--hidden", "16",
        "--emb-dim", "16",
        "--batch-size", "2",
        "--seed", "123",
        "--name", "_test_repro_1",
        "--data-dir", str(data_dir),
        "--no-save"
    ])
    res1 = run(args1)

    # Run 2
    args2 = parse([
        "--arch", "mlp",
        "--epochs", "2",
        "--hidden", "16",
        "--emb-dim", "16",
        "--batch-size", "2",
        "--seed", "123",
        "--name", "_test_repro_2",
        "--data-dir", str(data_dir),
        "--no-save"
    ])
    res2 = run(args2)

    # Cleanup any written reports from default location
    shutil.rmtree(Path("reports/_test_repro_1"), ignore_errors=True)
    shutil.rmtree(Path("reports/_test_repro_2"), ignore_errors=True)

    assert res1["loss"] == pytest.approx(res2["loss"])
    assert res1["accuracy"] == pytest.approx(res2["accuracy"])
    assert res1["macro_f1"] == pytest.approx(res2["macro_f1"])
    assert res1["confusion_matrix"] == res2["confusion_matrix"]
