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
- **Commit**: `573f858`

## Step 2b: Extended Sweep on Dev (Results)
- **Date / Time**: 2026-11-07
- **Actions**:
  - Ran extended MIN_RETRIEVAL grid `[0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]` on dev claims (`data/claim_level/dev_claims.jsonl`, N=63) for both clean passage checkpoint (`bilstm_attn_passage_s42.pt`) and LEAKY deployed checkpoint (`bilstm_attn.pt`).
  - Saved outputs:
    - Clean: `results/threshold_sweep_passage_ckpt_extended.csv` and `.png`
    - LEAKY: `results/threshold_sweep_LEAKY_deployed_ckpt_extended.csv` and `.png`
- **Commands Run**:
  - `.\.venv\Scripts\python -m ml.sweep_thresholds --checkpoint ml/checkpoints/bilstm_attn_passage_s42.pt --claims data/claim_level/dev_claims.jsonl --label "Passage-Clean-Extended" --mr-grid "0.0,0.02,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6" --out results/threshold_sweep_passage_ckpt_extended.csv --fig results/threshold_sweep_passage_ckpt_extended.png`
  - `.\.venv\Scripts\python -m ml.sweep_thresholds --checkpoint ml/checkpoints/bilstm_attn.pt --claims data/claim_level/dev_claims.jsonl --label "LEAKY-Deployed-Extended" --mr-grid "0.0,0.02,0.04,0.05,0.06,0.08,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.5,0.6" --out results/threshold_sweep_LEAKY_deployed_ckpt_extended.csv --fig results/threshold_sweep_LEAKY_deployed_ckpt_extended.png`
  - `.\.venv\Scripts\python -m pytest -q`
- **Key Outputs**:
  - **Clean Checkpoint**:
    - Default (MR=0.05, ST=0.60): Accuracy = 0.3492 (95% CI: [0.2381, 0.4603]), Macro-F1 = 0.3355 (95% CI: [0.2188, 0.4424]), Shares: SUP=0.556, REF=0.111, CONF=0.143, INS=0.190.
    - Best (MR=0.30, ST=0.55): Accuracy = 0.5238 (95% CI: [0.3968, 0.6349]), Macro-F1 = 0.4776 (95% CI: [0.3679, 0.5886]), Shares: SUP=0.444, REF=0.095, CONF=0.000, INS=0.460.
    - Grid position: The peak at MR=0.30 is an interior point on the extended MR grid [0.0, 0.60]. The top region forms a plateau between MR in [0.25, 0.30] and ST in [0.55, 0.70] with accuracy 0.5238 and macro-F1 ~0.475-0.478.
  - **LEAKY Deployed Checkpoint**:
    - Default (MR=0.05, ST=0.60): Accuracy = 0.6667 (95% CI: [0.5556, 0.7778]), Macro-F1 = 0.5932 (95% CI: [0.4962, 0.6916]), Shares: SUP=0.492, REF=0.032, CONF=0.048, INS=0.429.
    - Best (MR=0.08, ST=0.75): Accuracy = 0.7143 (95% CI: [0.6032, 0.8254]), Macro-F1 = 0.6324 (95% CI: [0.5281, 0.7397]), Shares: SUP=0.476, REF=0.048, CONF=0.000, INS=0.476.
    - Grid position: Interior point at MR=0.08, ST=0.75-0.85 forming a plateau.
- **Fallbacks**: None.



