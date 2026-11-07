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
- **Commit**: `539330e`

## Step 1: Match the Sweep to Backend
- **Date / Time**: 2026-11-07
- **Actions**:
  - Compared SHA-256 hashes of `data/sample/evidence.jsonl` (backend corpus) and `data/sample_passage/evidence.jsonl` (sweep corpus). Both are identical: `5B4DE55276C544E789902EB4C9CC20CD3F40D2518C64EC990C4EA455EEB65718`.
  - Verified retrieval mechanism: both sweep and backend (`main.py`) retrieve `TOP_K=3` passages per claim using `Retriever.search(claim, k)` and build identical item dictionaries `{"id": h["id"], "text": h["text"], "retrieval_score": h["score"], **p}` for `decide()`.
  - Added opt-in `--corpus` (default `EVIDENCE_PATH`), `--top-k` (default `TOP_K`), `--mr-grid`, and `--st-grid` comma-list arguments to `ml/sweep_thresholds.py`.
  - Added unit tests in `tests/test_sweep.py` covering grid parsing and CLI argument handling.
- **Commands Run**:
  - `Get-FileHash data/sample/evidence.jsonl, data/sample_passage/evidence.jsonl -Algorithm SHA256`
  - `.\.venv\Scripts\python -m pytest -q` (65 passed in 10.25s)
- **Key Outputs**:
  - Corpus files are identical byte-for-byte.
  - Sweep already retrieves top-3 matching backend config.
  - Grid overrides function cleanly with backwards-compatible defaults.
- **Fallbacks**: None needed (corpora match and TOP_K=3 matches).
- **Commit**: `2c74104`

## Step 2a: Extended Sweep Guard (Code)
- **Date / Time**: 2026-11-07
- **Actions**:
  - Added safety guard in `ml/sweep_thresholds.py` prohibiting full-grid sweeps when `--claims` target path contains "test".
  - Added unit test `test_run_sweep_refuses_test_set` in `tests/test_sweep.py`.
- **Commands Run**:
  - `.\.venv\Scripts\python -m pytest -q` (66 passed in 8.85s)
- **Key Outputs**:
  - Attempts to sweep full grid on test files raise `ValueError`.
- **Fallbacks**: None.


