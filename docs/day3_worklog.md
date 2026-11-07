# Day 3 Worklog

## Step 0: Initialisation and Workspace Cleanup
- **Date / Time**: 2026-11-07
- **Actions**:
  - Verified current branch: `day3-thresholds-demo`.
  - Ran `pytest -q`: 64 passed in 9.66s.
  - Deleted hand-made files `results/threshold_sweep_passage_ckpt_extended.csv` and `results/threshold_sweep_passage_ckpt_extended.png`.
  - Renamed `results/sweep_extended_log.txt` to `results/threshold_sweep_passage_ckpt_ci.txt`.
  - Initialised `docs/day3_worklog.md`.
- **Commands Run**:
  - `git branch --show-current`
  - `git status --short`
  - `.\.venv\Scripts\python -m pytest -q`
  - `Remove-Item results/threshold_sweep_passage_ckpt_extended.csv, results/threshold_sweep_passage_ckpt_extended.png`
  - `Move-Item results/sweep_extended_log.txt results/threshold_sweep_passage_ckpt_ci.txt`
- **Key Outputs**:
  - Test suite passes cleanly with 64 tests.
  - Tracked status clean before staging.
- **Fallbacks**: None.
