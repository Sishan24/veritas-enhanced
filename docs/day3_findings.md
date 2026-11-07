# Day 3 Findings: Threshold Sweeps, Refuted Diagnostic, and Honest Evaluation

## Pre-registration of Test Settings

In accordance with empirical best practices and to avoid test-set snooping or post-hoc threshold selection, the evaluation settings for the test set (`data/claim_level/test_claims.jsonl`, N=60) are pre-registered prior to running any test evaluations.

### Pre-registered Configurations

1. **Setting A (System Defaults)**:
   - `MIN_RETRIEVAL = 0.05`
   - `STANCE_THRESHOLD = 0.60`
   - *Rationale*: These are the existing production defaults in `backend/app/config.py`.

2. **Setting B (Dev-Best Thresholds)**:
   - `MIN_RETRIEVAL = 0.30`
   - `STANCE_THRESHOLD = 0.55`
   - *Rationale*: Dev-best point identified in Step 2 extended sweep on the clean passage-trained checkpoint (`ml/checkpoints/bilstm_attn_passage_s42.pt`) over `data/claim_level/dev_claims.jsonl` (N=63). Tie-breaking rule (lowest `MIN_RETRIEVAL`, then `STANCE_THRESHOLD` closest to 0.60) selected this point, which uniquely achieved peak dev Macro-F1 (0.4776) and peak dev Accuracy (0.5238).

### Planned Test Evaluations

The test set (`data/claim_level/test_claims.jsonl`) will be evaluated exactly once under the following protocol:
- **Clean Passage Checkpoint** (`ml/checkpoints/bilstm_attn_passage_s42.pt`):
  - Evaluated at **Setting A** (0.05, 0.60).
  - Evaluated at **Setting B** (0.30, 0.55).
- **LEAKY Deployed Checkpoint** (`ml/checkpoints/bilstm_attn.pt`):
  - Evaluated at **Setting A only** (0.05, 0.60) (labelled LEAKY).

No test data will be inspected or swept over a grid, and no other threshold combinations will be evaluated on the test set.
