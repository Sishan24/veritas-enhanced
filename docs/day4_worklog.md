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

