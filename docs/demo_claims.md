# VERITAS Demo Claims and Verification Protocol

This document lists verified demo-safe claims, backend configuration, cryptographic hashes, and documented operational limitations for live system demonstrations.

## Configuration & Environment Hashes

The backend runs the FastAPI application (`backend.app.main:app`) using the deployed pair-trained checkpoint and corpus:
- **`MIN_RETRIEVAL`**: `0.05`
- **`STANCE_THRESHOLD`**: `0.60`
- **`TOP_K`**: `3`
- **Deployed Checkpoint**: `ml/checkpoints/bilstm_attn.pt`
  - **SHA-256**: `797d9726a1d41bb96ca6cd809b966b5a634f0d5278d4677e3887c66be92d5b5f`
- **Evidence Corpus**: `data/sample/evidence.jsonl`
  - **SHA-256**: `5b4de55276c544e789902eb4c9cc20cd3f40d2518c64ec990c4ea455eeb65718`
  - Total passages indexed: 137

> [!NOTE]
> **Model Difference Notice**: The live demo backend uses `ml/checkpoints/bilstm_attn.pt` (the pair-trained deployed checkpoint). This checkpoint has seen the sample evidence passages during training (leakage). This is distinct from `ml/checkpoints/bilstm_attn_passage_s42.pt` (the clean passage-trained checkpoint evaluated for threshold sweeps and scientific analysis in the findings report).

---

## Morning-of-Demo Verification Command

To verify the API path and ensure all demo claims remain stable and safe on the morning of the demo, run:
```powershell
python scripts/demo_check.py
```
This script exercises all 30 candidate claims and 3 user-supplied conflicting evidence requests through `fastapi.testclient.TestClient` against `backend.app.main:app`.

---

## Demo-Safe Claims

A claim is designated **demo-safe** only if:
1. Exactly one claim is segmented by `extract_claims` (`count == 1`).
2. The verdict is identical across all 3 independent runs.
3. The model confidence is outside the ambiguity band: $|confidence - 0.60| \ge 0.05$ (confidence $< 0.55$ or $> 0.65$).

### 1. Verified Safe SUPPORTED Claims
| Claim | Source / Topic | Deployed Verdict | Confidence | Evidence IDs |
| :--- | :--- | :---: | :---: | :--- |
| `Paris is the capital of France.` | Dev set | **SUPPORTED** | 0.9937 | `France`, `France-UNSC`, `Duplantis-world-record` |
| `Sirius Corporation was founded in 2004.` | Dev set | **SUPPORTED** | 1.0000 | `Sirius`, `Horizon`, `Atlas` |
| `2023 was confirmed as the hottest year on record globally.` | Dev set | **SUPPORTED** | 0.9999 | `hottest-year-2023`, `CO2-record-2023`, `India-economy-rank` |
| `Madrid is the capital of Spain.` | Test set | **SUPPORTED** | 0.9993 | `Spain`, `Euro-2024`, `Champions-League-2024` |
| `Acme Corporation was founded in 1948.` | Test set | **SUPPORTED** | 0.9981 | `Acme`, `Horizon`, `Atlas` |
| `The US Federal Reserve raised interest rates to a 22-year high.` | Corpus facts | **SUPPORTED** | 1.0000 | `Fed-rates`, `US-national-debt`, `Amazon-drought-2023` |
| `Spain won the UEFA Euro 2024 championship.` | Corpus facts | **SUPPORTED** | 1.0000 | `Euro-2024`, `Spain`, `Champions-League-2024` |
| `Canada experienced its worst wildfire season on record in 2023.` | Corpus facts | **SUPPORTED** | 1.0000 | `Canada`, `hottest-year-2023`, `Amazon-drought-2023` |
| `Global CO2 emissions from fossil fuels reached a new record in 2023.` | Corpus facts | **SUPPORTED** | 0.9999 | `CO2-record-2023`, `COP28`, `Bitcoin-price` |

### 2. Verified Safe REFUTED Claims
Only 3 candidate claims qualified as demo-safe REFUTED claims under the deployed model:
| Claim | Source / Topic | Deployed Verdict | Confidence | Evidence IDs |
| :--- | :--- | :---: | :---: | :--- |
| `Marseille is the capital of France.` | Dev false variant | **REFUTED** | 0.9839 | `France`, `France-UNSC`, `Ukraine` |
| `Istanbul is the capital of Turkey.` | Dev false variant | **REFUTED** | 0.9742 | `Turkey`, `Ukraine`, `Germany` |
| `Gothenburg is the capital of Sweden.` | Test false variant | **REFUTED** | 0.9995 | `Sweden`, `Ukraine`, `Germany` |

*Notes on other REFUTED claims*: Most other refuted variants (such as `Sirius Corporation was founded in 2016.`, `Acme Corporation was founded in 1945.`, `The US Federal Reserve lowered interest rates to record lows in 2023.`, and `England won the UEFA Euro 2024 championship in Berlin.`) are misclassified as **SUPPORTED** (confidence > 0.99) by the deployed checkpoint due to high lexical overlap with the corpus passage.

### 3. Verified Safe INSUFFICIENT Claims
| Claim | Source / Topic | Deployed Verdict | Confidence | Evidence IDs |
| :--- | :--- | :---: | :---: | :--- |
| `Apple reached a market capitalisation of three trillion dollars.` | Test set | **INSUFFICIENT** | 1.0000 | Low retrieval scores / no match |
| `The European Space Agency launched a rover to explore Venus in 2024.` | Out-of-corpus | **INSUFFICIENT** | 1.0000 | Irrelevant retrieval suppressed |
| `The international committee approved new guidelines for lunar mining rights.` | Out-of-corpus | **INSUFFICIENT** | 1.0000 | Irrelevant retrieval suppressed |

---

## Limitations on REFUTED and CONFLICTING

### 1. REFUTED Failure Modes
- The deployed pair-trained model suffers from severe lexical overlap bias: when a false claim differs from a true corpus passage by only a few words (e.g. changing 2004 to 2016, or 'raised' to 'lowered'), the model attends heavily to the shared words and outputs $P(\text{SUPPORTED}) > 0.95$.
- As demonstrated in `scripts/refuted_diagnostic.py`, retrieval surfaces the contradicting passage in 95.2% of cases, but the model assigns $P(\text{REFUTED})$ as its top stance in only 5.0% of cases.
- **Demo Rule**: Present ONLY the three vetted safe REFUTED claims listed above (`Marseille`, `Istanbul`, and `Gothenburg`). Do not improvise new refuted claims live.

### 2. CONFLICTING Verdict Limitations
- The built-in corpus does not contain contradictory documents for the same fact.
- To demonstrate CONFLICTING, user-supplied evidence must be provided via the per-request `evidence` list.
- Under test conditions with two user-supplied passages (one supporting, one contradicting):
  - **Attempt 1 (Success - Demo Safe)**:
    - Claim: `"Ankara is the capital of Turkey."`
    - Passage 1: `"The capital city of Turkey is Ankara."` ($P(\text{SUPPORTED}) = 0.9967$)
    - Passage 2: `"The capital city of Turkey is Istanbul, according to historical maritime records."` ($P(\text{REFUTED}) = 0.8692$)
    - Verdict: **`CONFLICTING`** (Confidence: `0.9967`)
    - Label: *demo-only user-supplied evidence*
  - **Attempt 2 & 3 (Failures)**:
    - Attempt 2 (`Acme Corporation was founded in 1948.`): Both passages resulted in $P(\text{SUPPORTED}) \ge 0.70$, producing `SUPPORTED`.
    - Attempt 3 (`The Federal Reserve raised interest rates.`): Both passages resulted in $P(\text{REFUTED}) \ge 0.99$, producing `REFUTED`.
- **Demo Rule**: State plainly that CONFLICTING cannot be shown reliably across arbitrary claims. It functions successfully only on Attempt 1 (`Ankara is the capital of Turkey.`) under *demo-only user-supplied evidence*.
