"""Tests for ml/run_experiments.py."""
import csv
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ml.run_experiments import (
    execute_run,
    get_all_configs,
    main,
    make_run_name,
)


def test_run_name_uniqueness():
    configs = get_all_configs()
    names = [c["name"] for c in configs]
    assert len(names) == 96
    assert len(set(names)) == 96, "Duplicate run names detected!"

    # Verify expected naming convention: <arch>_h<hidden>_<split>_ms<seed>
    for c in configs:
        name = c["name"]
        assert name.startswith(c["arch"])
        assert f"_h{c['hidden']}_" in name
        assert f"_ms{c['model_seed']}" in name
        if c["split"] == "passage":
            assert f"_passage_ss{c['split_seed']}_" in name
        else:
            assert "_pair_" in name


def test_skip_completed_runs(tmp_path):
    out_dir = tmp_path / "runs"
    log_path = tmp_path / "results" / "run_log.csv"
    run_name = "test_run_completed"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text('{"accuracy": 0.8}', encoding="utf-8")

    mock_runner = MagicMock()
    cfg = {
        "name": run_name,
        "kind": "neural",
        "arch": "mlp",
        "hidden": 128,
        "split": "pair",
        "split_seed": None,
        "data_dir": "data/sample",
        "model_seed": 0,
    }

    status, secs, err = execute_run(cfg, str(out_dir), log_path, runner_fn=mock_runner)
    assert status == "SKIPPED"
    assert secs == 0.0
    mock_runner.assert_not_called()

    # Check log
    assert log_path.exists()
    rows = list(csv.reader(log_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2
    assert rows[1][0] == run_name
    assert rows[1][1] == "SKIPPED"


def test_refuse_overwrite_incomplete_dir(tmp_path):
    out_dir = tmp_path / "runs"
    log_path = tmp_path / "results" / "run_log.csv"
    run_name = "test_run_incomplete"
    run_dir = out_dir / run_name
    run_dir.mkdir(parents=True)
    # create some file but NOT metrics.json
    (run_dir / "partial.log").write_text("in progress...", encoding="utf-8")

    mock_runner = MagicMock()
    cfg = {
        "name": run_name,
        "kind": "neural",
        "arch": "mlp",
        "hidden": 128,
        "split": "pair",
        "split_seed": None,
        "data_dir": "data/sample",
        "model_seed": 0,
    }

    status, secs, err = execute_run(cfg, str(out_dir), log_path, runner_fn=mock_runner)
    assert "REFUSED" in status
    mock_runner.assert_not_called()


def test_failed_run_retried_once_and_logged(tmp_path):
    out_dir = tmp_path / "runs"
    log_path = tmp_path / "results" / "run_log.csv"
    run_name = "test_run_fail"

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "Out of memory error"
    mock_proc.stdout = ""
    mock_runner = MagicMock(return_value=mock_proc)

    cfg = {
        "name": run_name,
        "kind": "neural",
        "arch": "mlp",
        "hidden": 128,
        "split": "pair",
        "split_seed": None,
        "data_dir": "data/sample",
        "model_seed": 0,
    }

    status, secs, err = execute_run(cfg, str(out_dir), log_path, runner_fn=mock_runner)
    assert status == "FAILED"
    assert "Out of memory" in err
    # Assert retried at most once (2 calls total)
    assert mock_runner.call_count == 2

    # Check log
    rows = list(csv.reader(log_path.read_text(encoding="utf-8").splitlines()))
    assert len(rows) == 2
    assert rows[1][0] == run_name
    assert rows[1][1] == "FAILED"


def test_dry_run():
    cfgs = main(["--dry-run"])
    assert len(cfgs) == 96


def test_model_seeds_flag():
    # Opt-in subset of seeds
    cfgs = main(["--dry-run", "--model-seeds", "0", "1"])
    assert len(cfgs) == 64
    assert set(c["model_seed"] for c in cfgs) == {0, 1}
    # Ensure run names and properties still follow conventions
    for c in cfgs:
        assert f"_ms{c['model_seed']}" in c["name"]

    # Single seed
    cfgs_single = main(["--dry-run", "--model-seeds", "42"])
    assert len(cfgs_single) == 32
    assert set(c["model_seed"] for c in cfgs_single) == {42}

    # Default remains all 3 seeds (0, 1, 2)
    cfgs_default = main(["--dry-run"])
    assert len(cfgs_default) == 96
    assert set(c["model_seed"] for c in cfgs_default) == {0, 1, 2}
