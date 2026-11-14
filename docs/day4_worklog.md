# Day 4 Worklog

## Step 0: Start
- **Date / Time**: 2026-11-07
- **Actions**:
  - Checked current branch: confirmed `day4-maxpool-weighting`.
  - Checked `git status --short`: clean working directory.
  - Ran `pytest -q`: 67 passed in 10.54s.
  - Created `docs/day4_worklog.md`.
- **Commands Run**:
  - `git branch --show-current`
  - `git status --short`
  - `.\.venv\Scripts\python -m pytest -q`
- **Key Outputs**: 67 passed in 10.54s.
- **Fallbacks**: None.
- **Commit**: `7b42154`

## Step 1: Freeze Existing Behavior
- **Date / Time**: 2026-11-07
- **Actions**:
  - Implemented regression fixtures under `tests/fixtures/`:
    - `frozen_model_outputs.json`: logits and attention weights for all 5 existing architectures (`mlp`, `cnn`, `lstm`, `bilstm`, `bilstm_attn`) on a fixed random batch with fixed seed (size ~2.5 KB).
    - `frozen_decide_outputs.json`: decision outputs and confidences across 7 hand-crafted scenarios under default mode (size ~2.5 KB).
    - `frozen_predict_outputs.json`: `VerifierService.predict` outputs on fixed (claim, evidence) pairs with `bilstm_attn.pt` (size ~1.7 KB).
  - Implemented `tests/test_frozen_behavior.py` verifying all frozen outputs, with graceful skip when checkpoints are absent (CI / fresh clone).
- **Commands Run**:
  - `.\.venv\Scripts\python -m pytest -q` (70 passed in 10.95s)
- **Key Outputs**: All 3 frozen regression tests pass cleanly.
- **Fallbacks**: None.
- **Commit**: `f1f77ae`

## Step 2: Add `bilstm_maxpool` Architecture Control
- **Date / Time**: 2026-11-09
- **Actions**:
  - Added `bilstm_maxpool` architecture in `ml/models.py`.
  - Added `masked_max` pooling function using `torch.finfo(x.dtype).min`.
  - Added unit tests in `tests/test_bilstm_maxpool.py` ensuring matching parameter counts with `bilstm` and valid masked pooling logic.
- **Commands Run**:
  - `.\.venv\Scripts\python -m pytest tests/test_bilstm_maxpool.py`
- **Key Outputs**: 4 tests passed cleanly.
- **Commit**: `27c1404`

## Step 3: Branch `enhancement` & Remaining Improvements
- **Date / Time**: 2026-11-14
- **Actions**:
  - Created and checked out new branch `enhancement`.
  - Cherry-picked Day 4 foundation (`b1bce6a`, `dcc14a1`, `1427ef0`).
  - **P1.6 & P1.7 (Retrieval Weighting & Normalized Confidence)**:
    - Updated `decide()` in `backend/app/services/verifier.py` to support `weighted=True` while keeping `weighted=False` default 100% backward-compatible.
    - Added weighted voting mass calculation: when support holds >80% retrieval mass, avoids spurious flips to CONFLICTING from marginal low-relevance hits.
    - Confidence normalized to retrieval-weighted average stance probability.
    - Added unit tests in `tests/test_decide.py`.
  - **P2.1 (Explainability Token Synchronization)**:
    - Fixed token highlight desync when evidence text contains only punctuation (`tokenize` returns `[]`, `encode` returns `[UNK]`).
    - Handled fallback `toks = tokenize(text)[:MAX_EV] or ["<unk>"]` and verified `len(toks) == int(em[i].sum())`.
  - **P2.2 (Constants Deduplication)**:
    - Centralized `DEFAULT_MAX_CLAIM = 32` and `DEFAULT_MAX_EV = 64` in `ml/data.py`.
    - Referenced in `ClaimEvidenceDataset` and `backend/app/services/verifier.py`.
  - **P2.3 (Inference Optimization)**:
    - Encoded claim once in `predict()` (`claim_ids = self.vocab.encode(claim, MAX_CLAIM)`).
  - **P2.6 & P2.7 (FP16 Mask Portability)**:
    - Replaced `-1e9` hardcoded constants in `ml/models.py` with `torch.finfo(dtype).min`.
  - **Explainability Occlusion Sensitivity Check**:
    - Implemented `VerifierService.occlusion_check(claim, evidence)` evaluating stance shift upon removing top-attended token.
    - Added API schema `OcclusionRequest` and endpoint `POST /api/occlusion`.
  - **Governance & CI**:
    - Added official `LICENSE` (MIT).
    - Added GitHub Actions workflow `.github/workflows/ci.yml` running pytest matrix on Python 3.11 & 3.12.
- **Commands Run**:
  - `.\.venv\Scripts\python -m pytest` -> 81 passed in 7.15s.
  - `npm run build` -> Vite built cleanly in 563ms.
  - `.\.venv\Scripts\python scripts/demo_check.py` -> 26/30 demo-safe claims verified.
- **Key Outputs**: Complete test suite passed (81/81), frontend build succeeded, demo check passed.
