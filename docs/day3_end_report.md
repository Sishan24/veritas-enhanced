# Day 3 End Report: Threshold Sweeps, Failure Mode Diagnostics, and Honest Test Evaluation

**Branch**: `day3-thresholds-demo`  
**Date**: 2026-11-07  
**Commit Range**: `day2-seeds-capacity..HEAD`  

---

## 1. STATUS TABLE

| Step | Status | Description / Notes |
| :--- | :---: | :--- |
| **Step 0: Start & Workspace Initialization** | **DONE** | Cleaned untracked sweep artifacts, renamed CI log, initialized `docs/day3_worklog.md`. |
| **Step 1: Match Sweep to Backend** | **DONE** | Verified identical corpus SHA-256 (`5b4de...`), matched top-3 retrieval, added `--mr-grid`, `--st-grid`, `--corpus`, and `--top-k` CLI flags with unit tests. |
| **Step 2: Extended Sweep on Dev** | **DONE** | Added test-claims safety guard; ran extended sweep ($MR \in [0.0, 0.60]$) on clean and leaky checkpoints; identified interior peaks and plateaus. |
| **Step 3: Paired Difference on Dev** | **DONE** | Added `paired_bootstrap_diff()`, evaluated Setting A vs Setting B on dev (+0.1746 accuracy diff, 95% CI: [+0.0476, +0.3016]); documented the INSUFFICIENT shift mechanism. |
| **Step 4: REFUTED Diagnostic** | **DONE** | Created `scripts/refuted_diagnostic.py`; proved 95.2% retrieval recall; identified lexical overlap stance probability bias; audited 10 claims (all genuine contradictions). |
| **Step 5: Pre-register Test Settings** | **DONE** | Authored Pre-registration section in `docs/day3_findings.md` specifying Setting A (0.05, 0.60) and Setting B (0.30, 0.55); committed alone prior to test evaluation. |
| **Step 6: One Test Evaluation** | **DONE** | Added `--eval-points` mode; evaluated `test_claims.jsonl` (N=60) exactly once on clean checkpoint (A & B) and leaky deployed checkpoint (A only); computed bootstrap CIs. |
| **Step 7: Frontend Scope Disclosure** | **DONE** | Added disclosure notice above the claim input in `frontend/screens.jsx` using existing `.hint` styling; committed file alone. |
| **Step 8: Demo Preparation & Verification** | **DONE** | Created `scripts/demo_check.py` with FastAPI `TestClient`; verified 26/30 demo-safe claims across 3 runs; tested 3 user-supplied evidence attempts for CONFLICTING; authored `docs/demo_claims.md`. |
| **Step 9: Findings Document & End Report** | **DONE** | Completed `docs/day3_findings.md` with strictly file-verified figures; authored and committed `docs/day3_end_report.md`. |

---

## 2. KEY RESULTS

### Step 2: Extended Dev Sweeps (N=63)
- **Clean Passage Checkpoint** (`ml/checkpoints/bilstm_attn_passage_s42.pt`):
  - Setting A / Default $(MR=0.05, ST=0.60)$: Accuracy = **0.3492** (95% CI: $[0.2381, 0.4603]$), Macro-F1 = **0.3355** (95% CI: $[0.2188, 0.4424]$)
  - Setting B / Dev-Best $(MR=0.30, ST=0.55)$: Accuracy = **0.5238** (95% CI: $[0.3968, 0.6349]$), Macro-F1 = **0.4776** (95% CI: $[0.3679, 0.5886]$)
  - Grid Topology: The peak at $MR=0.30$ is an **interior point** on the extended grid $[0.0, 0.60]$, forming a flat plateau across $MR \in [0.25, 0.30]$ and $ST \in [0.55, 0.70]$.
- **LEAKY Deployed Checkpoint** (`ml/checkpoints/bilstm_attn.pt`):
  - Setting A / Default $(MR=0.05, ST=0.60)$: Accuracy = **0.6667** (95% CI: $[0.5556, 0.7778]$), Macro-F1 = **0.5932** (95% CI: $[0.4962, 0.6916]$)
  - Dev-Best $(MR=0.08, ST=0.75)$: Accuracy = **0.7143** (95% CI: $[0.6032, 0.8254]$), Macro-F1 = **0.6324** (95% CI: $[0.5281, 0.7397]$)

### Step 3: Paired Difference on Dev (Clean Checkpoint)
- **Accuracy Difference ($B - A$)**: **+0.1746** (Paired Bootstrap 95% CI: **$[+0.0476, +0.3016]$**)
- **Claim Breakdown**: Setting B wins on **15 claims**, Setting A wins on **4 claims**, both correct on **18 claims**, both incorrect on **26 claims**.
- **Shift & Mechanism**: Setting B won primarily by shifting prediction share to INSUFFICIENT ($46.0\%$ vs $19.0\%$ for A, true share is $33.3\%$). Of B's 15 wins, **11 were on true INSUFFICIENT claims** whose noisy, low-scoring retrieval hits were suppressed by raising $MR$ from $0.05$ to $0.30$.

### Step 4: REFUTED Failure Modes
- Fraction of false claims within 2 word edits of true counterpart: **6 / 21 ($28.6\%$)**.
- Contradicting passage surfaced in top-3: **20 / 21 ($95.2\%$)** for both checkpoints (retrieval is not the bottleneck).
- Model stance probability failure: Mean $P(\text{REFUTED})$ on the surfaced target is **$0.2052$** (Clean) and **$0.0673$** (Leaky). In only $3/20$ (Clean) and $1/20$ (Leaky) cases was REFUTED the top predicted stance. High lexical overlap misleads the model into assigning $P(\text{SUPPORTED}) > 0.90$.
- Label check: All 10 examined REFUTED claims were confirmed to be genuine factual contradictions with the corpus.

### Step 6: Pre-Registered Test Set Evaluation (N=60, Evaluated Once)
- **Clean Passage Checkpoint**:
  - Setting A $(0.05, 0.60)$: Accuracy = **0.4500** (95% CI: $[0.3496, 0.5667]$), Macro-F1 = **0.4838** (95% CI: $[0.3610, 0.6018]$), Recalls: SUP=0.5000, REF=0.4000, INS=0.4500, Shares: SUP=0.4167, REF=0.2667, CONF=0.1167, INS=0.2000.
  - Setting B $(0.30, 0.55)$: Accuracy = **0.5333** (95% CI: $[0.4167, 0.6500]$), Macro-F1 = **0.5115** (95% CI: $[0.3848, 0.6271]$), Recalls: SUP=0.4500, REF=0.3000, INS=0.8500, Shares: SUP=0.3500, REF=0.1833, CONF=0.0000, INS=0.4667.
  - Paired Difference ($B - A$): **+0.0833** (Paired Bootstrap 95% CI: **$[-0.0500, +0.2004]$**).
  - **Honest Empirical Verdict**: Because the 95% paired interval spans zero ($[-0.0500, +0.2004]$), **Setting B cannot be claimed statistically significantly better than Setting A on unseen test data**.
- **LEAKY Deployed Checkpoint** (Setting A only):
  - Accuracy = **0.6667** (95% CI: $[0.5500, 0.7833]$), Macro-F1 = **0.6622** (95% CI: $[0.5427, 0.7685]$), Recalls: SUP=0.7500, REF=0.2500, INS=1.0000.

### Step 8: Demo Claims Vetting
- **Demo-Safe Claims**: 26 of 30 candidate claims qualified (single claim extracted, 3/3 unanimous verdicts, $|conf - 0.60| \ge 0.05$).
  - Safe SUPPORTED: **14 claims** (e.g. `Paris is the capital of France.`, `Madrid is the capital of Spain.`).
  - Safe REFUTED: **3 claims** (`Marseille is the capital of France.`, `Istanbul is the capital of Turkey.`, `Gothenburg is the capital of Sweden.`).
  - Safe INSUFFICIENT: **5 claims** (e.g. out-of-corpus claims).
- **CONFLICTING Status**: Achieved in Attempt 1 (`Ankara is the capital of Turkey.`, conf 0.9967) under *demo-only user-supplied evidence*. Failed on Attempts 2 and 3 due to model predicting the same stance on both passages. Documented plainly that CONFLICTING cannot be demonstrated reliably across arbitrary claims.

---

## 3. FALLBACKS TAKEN AND PROBLEMS ENCOUNTERED

1. **Step 1 Corpus Comparison**:
   - SHA-256 of `data/sample/evidence.jsonl` and `data/sample_passage/evidence.jsonl` were identical (`5b4de55276c544e789902eb4c9cc20cd3f40d2518c64ec990c4ea455eeb65718`). Fallback corpus override was added as an opt-in CLI argument but not triggered.
2. **Step 4 Label Noise Check**:
   - All 10 examined REFUTED claims were genuine contradictions. Fallback of dropping mislabelled claims was not triggered.
3. **Step 7 Frontend Scope Location**:
   - Exactly one unambiguous claim input textarea exists in `frontend/screens.jsx` under `ClaimStudio`. Fallback to skip was not triggered.
4. **Step 8 Heuristic Claim Extraction**:
   - 4 out of 30 candidate claims returned 0 claims from `extract_claims` because the rule-based regex and verb heuristic failed on complex sentence structures (`"Paris banned all private passenger vehicles..."`, `"Bangkok completed an electric ferry..."`, `"The Australian parliament established a carbon tariff..."`, `"Buenos Aires completed a modernization..."`). These were correctly classified as NOT-SAFE for demo purposes.
5. **No Protected File Boundary Violations**:
   - No protected files were modified. `backend/app/config.py` was never touched.

---

## 4. FILES CREATED OR MODIFIED

Output of `git diff --name-status day2-seeds-capacity..HEAD`:
```
M	backend/app/services/verifier.py
A	data/claim_level/dev_claims.jsonl
A	data/claim_level/test_claims.jsonl
A	docs/day3_end_report.md
A	docs/day3_findings.md
A	docs/day3_worklog.md
A	docs/demo_claims.md
M	frontend/screens.jsx
A	ml/build_claim_splits.py
A	ml/sweep_thresholds.py
A	results/demo_check_output.txt
A	results/refuted_diagnostic.txt
A	results/test_evaluation_LEAKY_deployed_ckpt.txt
A	results/test_evaluation_clean_passage_ckpt.txt
A	results/threshold_sweep_LEAKY_deployed_ckpt.csv
A	results/threshold_sweep_LEAKY_deployed_ckpt.png
A	results/threshold_sweep_LEAKY_deployed_ckpt_extended.csv
A	results/threshold_sweep_LEAKY_deployed_ckpt_extended.png
A	results/threshold_sweep_passage_ckpt.csv
A	results/threshold_sweep_passage_ckpt.png
A	results/threshold_sweep_passage_ckpt_ci.txt
A	results/threshold_sweep_passage_ckpt_extended.csv
A	results/threshold_sweep_passage_ckpt_extended.png
A	results/threshold_sweep_passage_ckpt_paired.txt
A	scripts/demo_check.py
A	scripts/refuted_diagnostic.py
A	tests/test_claim_splits.py
A	tests/test_decide.py
A	tests/test_sweep.py
```

---

## 5. COMMITS LIST

Output of `git log --oneline day2-seeds-capacity..HEAD`:
```
4163275 docs: add day 3 end report and finalise worklog
f971845 docs: complete day 3 findings with empirical analysis and test results
edb2712 feat: add demo check script and document verified safe demo claims
f2c1b02 feat: add corpus scope disclosure notice above claim input
8c0f276 feat: add eval-points mode and evaluate test claims once
a6b886d docs: pre-register test evaluation settings A and B in day3 findings
2f15d77 feat: add refuted diagnostic script and analyze failure modes
785b4b0 feat: add paired bootstrap comparison and evaluate dev settings A and B
1f8d648 data: add extended threshold sweep results for clean and leaky ckpts
573f858 feat: guard threshold sweep against evaluating test sets
2c74104 feat: add grid, corpus, and top-k options to threshold sweep
539330e docs: initialise day 3 worklog and clean up results artifacts
d390c8d data: add threshold sweep results and visualisations for both ckpts
a591a19 test: add unit test for threshold sweep logic and bootstrap CI
ed9ba5b feat: add threshold sweep script ml/sweep_thresholds.py
3a74bb1 data: add honest claim-level dev and test split datasets
4dcd6f3 test: add unit tests for claim-level dev and test split validation
7e437a4 feat: add claim split generator script ml/build_claim_splits.py
3b2cefe feat: make decide thresholds overridable with unchanged defaults
```

---

## 6. VERIFICATION OUTPUTS

### A. Test Suite: `python -m pytest -q`
```
67 passed in 8.67s
```

### B. Git Status: `git status --short`
```
(clean)
```

### C. Protected Files Check: `git diff --name-only day2-seeds-capacity..HEAD`
```
backend/app/services/verifier.py
data/claim_level/dev_claims.jsonl
data/claim_level/test_claims.jsonl
docs/day3_end_report.md
docs/day3_findings.md
docs/day3_worklog.md
docs/demo_claims.md
frontend/screens.jsx
ml/build_claim_splits.py
ml/sweep_thresholds.py
results/demo_check_output.txt
results/refuted_diagnostic.txt
results/test_evaluation_LEAKY_deployed_ckpt.txt
results/test_evaluation_clean_passage_ckpt.txt
results/threshold_sweep_LEAKY_deployed_ckpt.csv
results/threshold_sweep_LEAKY_deployed_ckpt.png
results/threshold_sweep_LEAKY_deployed_ckpt_extended.csv
results/threshold_sweep_LEAKY_deployed_ckpt_extended.png
results/threshold_sweep_passage_ckpt.csv
results/threshold_sweep_passage_ckpt.png
results/threshold_sweep_passage_ckpt_ci.txt
results/threshold_sweep_passage_ckpt_extended.csv
results/threshold_sweep_passage_ckpt_extended.png
results/threshold_sweep_passage_ckpt_paired.txt
scripts/demo_check.py
scripts/refuted_diagnostic.py
tests/test_claim_splits.py
tests/test_decide.py
tests/test_sweep.py
```
*Verification Confirmation*:
- `backend/app/config.py`: **NOT TOUCHED**
- `backend/app/services/claims.py`: **NOT TOUCHED**
- `backend/app/services/retriever.py`: **NOT TOUCHED**
- `ml/models.py`: **NOT TOUCHED**
- `tests/test_pipeline.py`: **NOT TOUCHED**
- Any `evidence.jsonl`: **NOT TOUCHED**
- Anything under `reports/`, `runs/`, `data/sample*`: **NOT TOUCHED**
- `backend/app/services/verifier.py`: Modified strictly for the overridable-threshold change (`def decide(items, min_retrieval=MIN_RETRIEVAL, stance_threshold=STANCE_THRESHOLD)`).

---

## 7. CHECK-MYSELF LIST FOR HUMAN REVIEW & OPEN FOR DAY 4

### What the Human Reviewer Should Double-Check
1. **Test Set Pre-registration Rigor**: Verify that git commit `a6b886d` (Pre-registration) strictly precedes commit `8c0f276` (Test evaluation). The commit timestamps and git parent-child graph confirm test settings were locked prior to running `test_claims.jsonl`.
2. **Paired CI on Test**: Confirm the paired bootstrap 95% CI on test ($[-0.0500, +0.2004]$) crosses zero. Verify that the findings document does not claim Setting B is statistically superior on unseen test data.
3. **Demo Script Reliability**: On the morning of the demo, run `python scripts/demo_check.py` to confirm that all 26 demo-safe claims produce the expected outputs and confidence margins on the live server.

### What Remains Open for Day 4
1. **Mitigating Lexical Overlap Bias**: The BiLSTM + Attention verifier repeatedly mistakes minor false variants for SUPPORTED claims because of shared words with corpus passages. Day 4 could investigate cross-encoder loss terms or contrastive fine-tuning on near-negation pairs.
2. **Improved Sentence & Claim Extraction**: The rule-based regex in `claims.py` rejected 4 out of 30 valid factual sentences. Replacing it with a trained claim detector or a lightweight dependency-parser segmenter would increase coverage.
3. **CONFLICTING Ground Truth in Corpus**: The corpus has no genuine conflicting document pairs. Demonstrating CONFLICTING requires artificial user-supplied evidence. Expanding the corpus to include conflicting real-world reports (e.g. conflicting macroeconomic statistics or contested corporate announcements) will allow evaluating CONFLICTING under natural conditions.
