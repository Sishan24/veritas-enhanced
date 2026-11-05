"""Tests for ml/train.py seeding, flags, and output directory."""
import json
import shutil
from pathlib import Path
import pytest
from ml.train import parse, run


@pytest.fixture
def tiny_dataset(tmp_path):
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
    return data_dir


def test_identical_seed_produces_identical_metrics(tiny_dataset):
    # Run 1
    args1 = parse([
        "--arch", "mlp",
        "--epochs", "2",
        "--hidden", "16",
        "--emb-dim", "16",
        "--batch-size", "2",
        "--seed", "123",
        "--name", "_test_repro_1",
        "--data-dir", str(tiny_dataset),
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
        "--data-dir", str(tiny_dataset),
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


def test_default_out_dir_writes_to_reports(tiny_dataset):
    run_name = "_test_default_out"
    report_path = Path("reports") / run_name
    shutil.rmtree(report_path, ignore_errors=True)
    try:
        args = parse([
            "--arch", "mlp",
            "--epochs", "1",
            "--hidden", "16",
            "--emb-dim", "16",
            "--batch-size", "2",
            "--seed", "42",
            "--name", run_name,
            "--data-dir", str(tiny_dataset),
            "--no-save",
        ])
        assert args.out_dir == "reports"
        run(args)
        assert (report_path / "metrics.json").exists()
        assert (report_path / "history.json").exists()
    finally:
        shutil.rmtree(report_path, ignore_errors=True)


def test_custom_out_dir_writes_to_specified_dir(tmp_path, tiny_dataset):
    custom_out = tmp_path / "custom_runs"
    run_name = "_test_custom_out"
    args = parse([
        "--arch", "mlp",
        "--epochs", "1",
        "--hidden", "16",
        "--emb-dim", "16",
        "--batch-size", "2",
        "--seed", "42",
        "--name", run_name,
        "--out-dir", str(custom_out),
        "--data-dir", str(tiny_dataset),
        "--no-save",
    ])
    assert args.out_dir == str(custom_out)
    run(args)
    assert (custom_out / run_name / "metrics.json").exists()
    assert (custom_out / run_name / "history.json").exists()
    assert not (Path("reports") / run_name).exists()
