# Data

| Folder | What |
| --- | --- |
| `sample/` | Synthetic smoke-test data (`python -m ml.make_sample_data`). Proves the pipeline runs; **not** a research result. |
| `raw/` | Put downloaded datasets here (git-ignored). |
| `processed/` | Output of `ml.prepare_fever` (git-ignored). |

## Real data: FEVER
1. Download `train.jsonl`, `shared_task_dev.jsonl` and `wiki-pages.zip` from https://fever.ai/dataset/fever.html into `data/raw/` (unzip the wiki pages to `data/raw/wiki-pages/`).
2. Convert:
```powershell
python -m ml.prepare_fever --claims data/raw/train.jsonl --wiki data/raw/wiki-pages --out data/processed/train.jsonl --limit 50000
python -m ml.prepare_fever --claims data/raw/shared_task_dev.jsonl --wiki data/raw/wiki-pages --out data/processed/dev.jsonl --limit 5000
```
3. Create `data/processed/test.jsonl` (e.g. split off part of dev), then train: `python -m ml.train --data-dir data/processed`.

Other datasets from the project plan: LIAR (statement-level labels, no evidence) and FEVEROUS (adds table evidence). Both need their own converters to the `{claim, evidence, label}` format.
