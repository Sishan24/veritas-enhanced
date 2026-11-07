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

---

## Extended Threshold Sweeps on Dev Set (N=63)

Grid sweeps were executed over `data/claim_level/dev_claims.jsonl` (21 SUPPORTED, 21 REFUTED, 21 INSUFFICIENT) across an extended `MIN_RETRIEVAL` grid of `[0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]` and `STANCE_THRESHOLD` grid of `[0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]`.

### Summary Comparison Table (Dev Set)

| Checkpoint | Setting | MIN_RETRIEVAL | STANCE_THRESHOLD | Dev Accuracy (95% CI) | Dev Macro-F1 (95% CI) | Predicted Shares (SUP / REF / CONF / INS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Clean Passage** (`s42.pt`) | Default (A) | 0.05 | 0.60 | 0.3492 [0.2381, 0.4603] | 0.3355 [0.2188, 0.4424] | 0.556 / 0.111 / 0.143 / 0.190 |
| **Clean Passage** (`s42.pt`) | Best (B) | 0.30 | 0.55 | 0.5238 [0.3968, 0.6349] | 0.4776 [0.3679, 0.5886] | 0.444 / 0.095 / 0.000 / 0.460 |
| **LEAKY Deployed** (`attn.pt`) | Default (A) | 0.05 | 0.60 | 0.6667 [0.5556, 0.7778] | 0.5932 [0.4962, 0.6916] | 0.492 / 0.032 / 0.048 / 0.429 |
| **LEAKY Deployed** (`attn.pt`) | Best | 0.08 | 0.75 | 0.7143 [0.6032, 0.8254] | 0.6324 [0.5281, 0.7397] | 0.476 / 0.048 / 0.000 / 0.476 |

### Metric Surface and Grid Topology
- **Clean Checkpoint**: On the initial grid bounded at $MR \le 0.30$, the peak appeared on the boundary edge. With the extended grid up to $0.60$, the optimum at $(MR=0.30, ST=0.55)$ is an **interior point**. The surface forms a flat plateau across $MR \in [0.25, 0.30]$ and $ST \in [0.55, 0.70]$ with accuracy hovering at $0.5238$ and Macro-F1 at $0.475 - 0.478$.
- **LEAKY Deployed Checkpoint**: The optimal point $(MR=0.08, ST=0.75)$ is also clearly interior, forming a plateau across $ST \in [0.75, 0.85]$. Note that the deployed defaults $(0.05, 0.60)$ are already very close to the dev optimum ($0.6667$ vs $0.7143$, gap of $+0.0476$).

---

## Paired Difference Analysis on Dev (Clean Checkpoint)

A paired bootstrap analysis (1,000 resamples, fixed seed 42) was conducted comparing Setting B $(0.30, 0.55)$ against Setting A $(0.05, 0.60)$ on the identical 63 dev claims:
- **Observed Accuracy Difference ($B - A$)**: $+0.1746$
- **Paired Bootstrap 95% CI**: $[+0.0476, +0.3016]$
- **Claim-level Performance**:
  - Setting B wins (correct under B, incorrect under A): **15 claims**
  - Setting A wins (correct under A, incorrect under B): **4 claims**
  - Both correct: **18 claims**
  - Both incorrect: **26 claims**
- **Statistical Significance on Dev**: Because the 95% paired bootstrap interval $[+0.0476, +0.3016]$ strictly excludes zero, Setting B achieves a statistically significant accuracy improvement over Setting A on the dev distribution.

### The INSUFFICIENT-Shift Caveat

Analysis of the 15 claims where Setting B won reveals the underlying mechanism:
- **Class breakdown of B's 15 wins**:
  - True INSUFFICIENT: **11 claims**
  - True SUPPORTED: **3 claims**
  - True REFUTED: **1 claim**
- **Class breakdown of A's 4 wins**:
  - True SUPPORTED: **4 claims**
- **Mechanism**: The true dev set is balanced ($1/3$ or $33.3\%$ per class). Setting A severely under-predicted INSUFFICIENT ($19.0\%$) and over-predicted SUPPORTED ($55.6\%$) because low-scoring irrelevant retrieval hits ($0.05 \le \text{score} < 0.30$) were accepted by `decide()` and triggered false positive stances.
- Setting B raised `MIN_RETRIEVAL` to $0.30$, suppressing these weak hits and increasing the predicted INSUFFICIENT share to $46.0\%$ (recall of INSUFFICIENT rose from $28.6\%$ to $81.0\%$).
- **Plain Conclusion**: Setting B wins on dev primarily by predicting INSUFFICIENT more aggressively, trading off 4 SUPPORTED claims to correctly reject 11 noisy retrieval instances.

---

## REFUTED Diagnostic and Failure Mode Analysis

Both checkpoints exhibit critically depressed REFUTED recall on dev ($9.5\%$, 2 of 21 claims). `scripts/refuted_diagnostic.py` evaluated all 21 false claims against the corpus:

### Quantitative Findings
1. **Edit Distance to True Counterpart**:
   - $6 / 21$ ($28.6\%$) of false claims are within 2 word edits of their true counterpart (e.g. changing capital city names or founding years).
2. **Retrieval Performance**:
   - The contradicting passage is surfaced in the top-3 hits in **$20 / 21$ cases ($95.2\%$)** for both checkpoints. Retrieval failure is **not** the bottleneck.
3. **Model Stance Probability**:
   - **Clean Checkpoint**: Mean $P(\text{REFUTED})$ on the surfaced target passage is only **$0.2052$** (range: $0.0000$ to $0.9994$). The model assigns REFUTED as its highest stance in only **$3 / 20$ cases ($15.0\%$)**. In 11 cases, high lexical overlap causes the model to assign high $P(\text{SUPPORTED})$ (false positive).
   - **LEAKY Deployed Checkpoint**: Mean $P(\text{REFUTED})$ on the surfaced target passage is only **$0.0673$** (range: $0.0000$ to $0.9094$). It places REFUTED as highest in only **$1 / 20$ cases ($5.0\%$)**.
4. **Final `decide()` Verdicts on 21 REFUTED Claims**:
   - Clean Checkpoint: 2 REFUTED ($9.5\%$), 11 SUPPORTED ($52.4\%$), 4 CONFLICTING ($19.0\%$), 4 INSUFFICIENT ($19.0\%$).
   - LEAKY Deployed Checkpoint: 2 REFUTED ($9.5\%$), 12 SUPPORTED ($57.1\%$), 1 CONFLICTING ($4.8\%$), 6 INSUFFICIENT ($28.6\%$).

### Label Quality Check
Ten randomly audited REFUTED claims were evaluated side-by-side against their target corpus passage:
- In all 10 cases, the corpus passage directly and unambiguously contradicts the claim factually.
- No mislabelled claims were detected in the dev split. The poor recall is caused entirely by model stance bias under high lexical overlap, not label corruption.

---

## Pre-Registered Test Set Evaluation (N=60)

The pre-registered configurations were evaluated exactly once on `data/claim_level/test_claims.jsonl` (20 SUPPORTED, 20 REFUTED, 20 INSUFFICIENT).

### Test Performance Table

| Checkpoint | Setting | Accuracy (95% CI) | Macro-F1 (95% CI) | Recall SUP | Recall REF | Recall INS | Shares (SUP / REF / CONF / INS) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Passage** (`s42.pt`) | Setting A (0.05, 0.60) | 0.4500 [0.3496, 0.5667] | 0.4838 [0.3610, 0.6018] | 0.5000 (10/20) | 0.4000 (8/20) | 0.4500 (9/20) | 0.4167 / 0.2667 / 0.1167 / 0.2000 |
| **Clean Passage** (`s42.pt`) | Setting B (0.30, 0.55) | 0.5333 [0.4167, 0.6500] | 0.5115 [0.3848, 0.6271] | 0.4500 (9/20) | 0.3000 (6/20) | 0.8500 (17/20) | 0.3500 / 0.1833 / 0.0000 / 0.4667 |
| **LEAKY Deployed** (`attn.pt`) | Setting A (0.05, 0.60) [LEAKY] | 0.6667 [0.5500, 0.7833] | 0.6622 [0.5427, 0.7685] | 0.7500 (15/20) | 0.2500 (5/20) | 1.0000 (20/20) | 0.3833 / 0.0833 / 0.1167 / 0.4167 |

### Paired Comparison on Test (Setting B minus Setting A, Clean Checkpoint)
- **Observed Accuracy Difference**: $+0.0833$ ($+5$ claims net)
- **Paired Bootstrap 95% CI**: **$[-0.0500, +0.2004]$**
- **Claim Outcomes**:
  - Setting B wins: 11 claims
  - Setting A wins: 6 claims
  - Both correct: 21 claims
  - Both incorrect: 22 claims

> [!IMPORTANT]
> **Statistical Significance Ruling**: The paired 95% bootstrap confidence interval on the unseen test set spans zero ($[-0.0500, +0.2004]$). In accordance with the pre-registered decision rule, **Setting B cannot be claimed to be statistically significantly better than Setting A on unseen test data**. While Setting B achieved higher point accuracy ($53.3\%$ vs $45.0\%$), the difference is within expected sampling variation on a 60-claim test set.

---

## Limitations

1. **Sample Size**: Each evaluation set contains only 20 claims per class (63 total on dev, 60 on test). This sample size yields wide 95% confidence intervals ($\pm 11-12$ percentage points), making modest true effect sizes statistically undetectable.
2. **Single Checkpoint Seed**: Analysis of the clean passage-trained model is restricted to a single seed (seed 42). Multi-seed training variance was not assessed at the claim level.
3. **Small Evidence Corpus**: The sample evidence corpus comprises only 137 passages. In larger retrieval spaces (thousands to millions of documents), TF-IDF score distributions will shift substantially.
4. **Coupled Hyperparameters**: Decision thresholds (`MIN_RETRIEVAL`, `STANCE_THRESHOLD`) are tightly coupled to the specific model architecture (BiLSTM + Attention) and retriever (TF-IDF). They do not generalize directly to denser embedding models or different corpus distributions.
