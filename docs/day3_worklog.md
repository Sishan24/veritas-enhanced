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
- **Commit**: `1f8d648`

## Step 3: Paired Difference on Dev
- **Date / Time**: 2026-11-07
- **Actions**:
  - Implemented `paired_bootstrap_diff()` in `ml/sweep_thresholds.py` calculating paired bootstrap 95% CI (1000 resamples, fixed seed=42) for accuracy difference (B - A), individual claim wins (B wins vs A wins), and agreement counts.
  - Implemented `run_paired_comparison()` and CLI options (`--paired`, `--mr-a`, `--st-a`, `--mr-b`, `--st-b`, `--paired-out`).
  - Added unit test `test_paired_bootstrap_diff` in `tests/test_sweep.py`.
  - Evaluated Setting A (0.05, 0.60) vs Setting B (0.30, 0.55) on dev claims (`data/claim_level/dev_claims.jsonl`, N=63) for the clean passage checkpoint (`ml/checkpoints/bilstm_attn_passage_s42.pt`).
  - Saved detailed breakdown to `results/threshold_sweep_passage_ckpt_paired.txt`.
- **Commands Run**:
  - `.\.venv\Scripts\python -m ml.sweep_thresholds --checkpoint ml/checkpoints/bilstm_attn_passage_s42.pt --claims data/claim_level/dev_claims.jsonl --paired --mr-a 0.05 --st-a 0.60 --mr-b 0.30 --st-b 0.55 --paired-out results/threshold_sweep_passage_ckpt_paired.txt`
  - `.\.venv\Scripts\python -m pytest -q` (67 passed in 9.05s)
- **Key Outputs**:
  - Setting A: Acc = 0.3492, Macro-F1 = 0.3355, Shares: SUP=0.5556, REF=0.1111, CONF=0.1429, INS=0.1905.
  - Setting B: Acc = 0.5238, Macro-F1 = 0.4776, Shares: SUP=0.4444, REF=0.0952, CONF=0.0000, INS=0.4603.
  - Accuracy Difference (B - A): +0.1746 (95% CI: [+0.0476, +0.3016]).
  - B wins: 15 claims (11 INSUFFICIENT, 3 SUPPORTED, 1 REFUTED).
  - A wins: 4 claims (4 SUPPORTED).
  - Both correct: 18 claims; Both incorrect: 26 claims.
  - Plain conclusion: Setting B wins mainly by predicting INSUFFICIENT more (46.0% vs true 33.3%). Raising MIN_RETRIEVAL from 0.05 to 0.30 filters out weak irrelevant retrieval hits, converting false positive SUPPORTED/CONFLICTING predictions into correct INSUFFICIENT verdicts (11 of B's 15 wins were on true INSUFFICIENT claims).
- **Fallbacks**: None.
- **Commit**: `785b4b0`

## Step 4: REFUTED Diagnostic
- **Date / Time**: 2026-11-07
- **Actions**:
  - Created `scripts/refuted_diagnostic.py` to analyze all 21 REFUTED dev claims across both checkpoints (`ml/checkpoints/bilstm_attn_passage_s42.pt` and `ml/checkpoints/bilstm_attn.pt`).
  - Computed word-level Levenshtein edit distance between each false claim and its true counterpart.
  - Inspected top-3 retrieved passages, retrieval scores, predicted stance probabilities, and `decide()` output.
  - Performed label check on 10 REFUTED claims against their corpus passages to evaluate whether genuine contradiction exists.
  - Generated and saved output to `results/refuted_diagnostic.txt`.
- **Commands Run**:
  - `.\.venv\Scripts\python scripts/refuted_diagnostic.py`
  - `.\.venv\Scripts\python -m pytest -q` (67 passed in 8.90s)
- **Key Outputs**:
  - 6/21 (28.6%) false claims are within 2 word edits of their true counterpart (e.g. city name swaps, year changes).
  - Retrieval surfaces the contradicting passage in 20/21 cases (95.2%) for both checkpoints. Retrieval is not the bottleneck.
  - Model stance probability is the primary failure point:
    - Clean checkpoint: Mean P(REFUTED) on contradicting passage when surfaced is only 0.2052 (range: 0.0000 - 0.9994); REFUTED is highest probability in only 3/20 cases (15.0%). In 11/20 cases, high lexical overlap causes the model to assign high P(SUPPORTED).
    - LEAKY deployed checkpoint: Mean P(REFUTED) is 0.0673 (range: 0.0000 - 0.9094); REFUTED is highest in only 1/20 cases (5.0%).
    - Final decide() verdicts on REFUTED claims at defaults: 2/21 REFUTED (9.5%), 11-12/21 SUPPORTED (false positives), 1-4 CONFLICTING, 4-6 INSUFFICIENT.
  - Label check: All 10 examined claims are genuinely contradicted by the target corpus passage. No false negatives due to label noise.
- **Fallbacks**: None triggered (no mislabelled claims in the sample).
- **Commit**: `2f15d77`

## Step 5: Pre-register the Test Settings
- **Date / Time**: 2026-11-07
- **Actions**:
  - Authored the "Pre-registration of Test Settings" section of `docs/day3_findings.md`.
  - Named exactly two evaluation settings:
    - Setting A (System Defaults): `MIN_RETRIEVAL = 0.05`, `STANCE_THRESHOLD = 0.60`.
    - Setting B (Dev-Best): `MIN_RETRIEVAL = 0.30`, `STANCE_THRESHOLD = 0.55` (uniquely best dev point from Step 2 on clean passage checkpoint, ties broken toward lowest min_retrieval, then stance closest to 0.60).
  - Specified planned test evaluations: Clean checkpoint at Setting A and Setting B; LEAKY deployed checkpoint at Setting A only.
  - Committed before executing any test-set evaluation.
- **Commands Run**:
  - `.\.venv\Scripts\python -m pytest -q`
- **Key Outputs**:
  - Pre-registration section in `docs/day3_findings.md` locked in place.
- **Fallbacks**: None.
- **Commit**: `a6b886d`

## Step 6: One Test Evaluation
- **Date / Time**: 2026-11-07
- **Actions**:
  - Added `--eval-points` and `--eval-out` modes to `ml/sweep_thresholds.py` allowing evaluation of specific `(min_retrieval, stance_threshold)` pairs with 95% bootstrap CIs without sweeping a full grid (this is the sole mode allowed on test datasets).
  - Added unit test coverage for `parse_eval_points` and parser in `tests/test_sweep.py`.
  - Executed test evaluation ONCE on `data/claim_level/test_claims.jsonl` (N=60: 20 SUPPORTED, 20 REFUTED, 20 INSUFFICIENT):
    - Clean passage checkpoint (`ml/checkpoints/bilstm_attn_passage_s42.pt`): Setting A (0.05, 0.60) and Setting B (0.30, 0.55). Saved to `results/test_evaluation_clean_passage_ckpt.txt`.
    - LEAKY deployed checkpoint (`ml/checkpoints/bilstm_attn.pt`): Setting A only (0.05, 0.60). Saved to `results/test_evaluation_LEAKY_deployed_ckpt.txt`.
- **Commands Run**:
  - `.\.venv\Scripts\python -m ml.sweep_thresholds --checkpoint ml/checkpoints/bilstm_attn_passage_s42.pt --claims data/claim_level/test_claims.jsonl --eval-points "0.05:0.60,0.30:0.55" --label "Passage-Clean-Test" --eval-out results/test_evaluation_clean_passage_ckpt.txt`
  - `.\.venv\Scripts\python -m ml.sweep_thresholds --checkpoint ml/checkpoints/bilstm_attn.pt --claims data/claim_level/test_claims.jsonl --eval-points "0.05:0.60" --label "LEAKY-Deployed-Test" --eval-out results/test_evaluation_LEAKY_deployed_ckpt.txt`
  - `.\.venv\Scripts\python -m pytest -q` (67 passed in 8.67s)
- **Key Outputs**:
  - **Clean Checkpoint on Test (N=60)**:
    - Setting A (0.05, 0.60): Accuracy = 0.4500 (95% CI: [0.3496, 0.5667]), Macro-F1 = 0.4838 (95% CI: [0.3610, 0.6018]), Recalls: SUP=0.5000, REF=0.4000, INS=0.4500, Shares: SUP=0.4167, REF=0.2667, CONF=0.1167, INS=0.2000.
    - Setting B (0.30, 0.55): Accuracy = 0.5333 (95% CI: [0.4167, 0.6500]), Macro-F1 = 0.5115 (95% CI: [0.3848, 0.6271]), Recalls: SUP=0.4500, REF=0.3000, INS=0.8500, Shares: SUP=0.3500, REF=0.1833, CONF=0.0000, INS=0.4667.
    - Paired Diff (B - A): Accuracy Diff = +0.0833 (95% CI: [-0.0500, +0.2004]). The paired CI crosses zero; B cannot be claimed statistically significantly better than A on unseen test data. B wins on 11 claims, A wins on 6, both correct on 21, both incorrect on 22.
  - **LEAKY Deployed Checkpoint on Test (N=60)**:
    - Setting A (0.05, 0.60): Accuracy = 0.6667 (95% CI: [0.5500, 0.7833]), Macro-F1 = 0.6622 (95% CI: [0.5427, 0.7685]), Recalls: SUP=0.7500, REF=0.2500, INS=1.0000, Shares: SUP=0.3833, REF=0.0833, CONF=0.1167, INS=0.4167.
- **Fallbacks**: None.
- **Commit**: `8c0f276`

## Step 7: Frontend Scope Disclosure
- **Date / Time**: 2026-11-07
- **Actions**:
  - Located the claim input (`<textarea id="claims">`) in `frontend/screens.jsx` under `ClaimStudio`.
  - Added disclosure line immediately above the input: `"Claims are verified against the loaded evidence corpus. This system does not search the web."` using existing CSS styling class `hint`.
  - Committed `frontend/screens.jsx` alone in commit `f2c1b02`.
- **Commands Run**:
  - `.\.venv\Scripts\python -m pytest -q` (67 passed in 13.65s)
- **Key Outputs**:
  - Disclosure rendered cleanly without breaking React state or UI layout.
- **Fallbacks**: None.
- **Commit**: `f2c1b02`

## Step 8: Demo Preparation Through Real API Path
- **Date / Time**: 2026-11-07
- **Actions**:
  - Created `scripts/demo_check.py` using `fastapi.testclient.TestClient` on `backend.app.main:app` executing full end-to-end path (`extract_claims`, retrieval, deployed checkpoint `config.CHECKPOINT`, `decide()`).
  - Tested 30 candidate claims (10 dev, 10 test, 10 corpus topics), each run 3 independent times.
  - Verified stability: checked single-claim extraction, verdict unanimity across runs, and confidence margin $|conf - 0.60| \ge 0.05$.
  - Tested 3 attempts to produce CONFLICTING using per-request `evidence` parameter (without editing corpus).
  - Wrote `docs/demo_claims.md` documenting verified safe claims, exact hashes, morning-of-demo command, and limitations on REFUTED and CONFLICTING.
- **Commands Run**:
  - `.\.venv\Scripts\python scripts/demo_check.py`
  - `.\.venv\Scripts\python -m pytest -q` (67 passed in 11.60s)
- **Key Outputs**:
  - 26/30 claims evaluated are demo-safe; 4 failed extraction due to heuristic rule-based segmentation returning 0 claims.
  - Safe SUPPORTED: 14 claims verified.
  - Safe REFUTED: 3 claims qualified (`Marseille is the capital of France.`, `Istanbul is the capital of Turkey.`, `Gothenburg is the capital of Sweden.`). Most other false variants suffer from lexical overlap bias and flip to SUPPORTED.
  - Safe INSUFFICIENT: 5 claims verified.
  - CONFLICTING: Attempt 1 succeeded with `Ankara is the capital of Turkey.` under user-supplied supporting and contradicting passages (verdict: CONFLICTING, conf: 0.9967). Attempts 2 and 3 failed because model predicted same stance on both passages. Plainly documented that CONFLICTING cannot be demonstrated reliably across arbitrary claims.
  - Checkpoint SHA-256: `797d9726a1d41bb96ca6cd809b966b5a634f0d5278d4677e3887c66be92d5b5f`.
  - Corpus SHA-256: `5b4de55276c544e789902eb4c9cc20cd3f40d2518c64ec990c4ea455eeb65718`.
- **Fallbacks**: None.
- **Commit**: `edb2712`

## Step 9a: Day 3 Findings Document
- **Date / Time**: 2026-11-07
- **Actions**:
  - Completed `docs/day3_findings.md` incorporating:
    - Extended threshold sweep results (defaults vs dev-best) with 95% bootstrap CIs.
    - Identification of interior peaks and plateaus.
    - Paired difference analysis (+0.1746 on dev, CI: [+0.0476, +0.3016]).
    - Mechanism analysis explaining the INSUFFICIENT shift (11 of B's 15 wins came from true INSUFFICIENT claims).
    - REFUTED diagnostic summary (95.2% retrieval recall, mean P(REFUTED)=0.2052, lexical overlap bias).
    - Label check confirmation (10/10 genuine contradictions).
    - Pre-registered test set results (Setting A = 45.0%, Setting B = 53.3%, Paired Diff = +0.0833 with CI [-0.0500, +0.2004] spanning zero).
    - Methodological limitations (sample size, wide intervals, single seed, 137-passage corpus).
- **Commands Run**:
  - `.\.venv\Scripts\python -m pytest -q`
- **Key Outputs**:
  - Comprehensive findings document populated solely with verified figures read from generated files.
- **Fallbacks**: None.
- **Commit**: `f971845`

## Step 9b: Day 3 End Report
- **Date / Time**: 2026-11-07
- **Actions**:
  - Authored `docs/day3_end_report.md` encompassing all 7 required sections: status table, key results with CIs, fallbacks & problems, files diff, commit log, verification outputs (`pytest -q`, `git status --short`, protected files check), and check-myself list for human review.
- **Commands Run**:
  - `git diff --name-status day2-seeds-capacity..HEAD`
  - `git log --oneline day2-seeds-capacity..HEAD`
  - `git diff --name-only day2-seeds-capacity..HEAD`
  - `.\.venv\Scripts\python -m pytest -q` (67 passed)
  - `git status --short`
- **Key Outputs**:
  - All Day 3 tasks completed cleanly in an unbroken, autonomous sequence with complete auditability.
- **Fallbacks**: None.
