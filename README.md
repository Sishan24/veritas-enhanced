# VERITAS: The Claim & Context Intelligence Engine

An evidence-aware deep learning framework for natural-language claim verification, stance detection and contextual contradiction analysis.
Paste an article, get each claim checked against evidence, and see **why** the model decided (word-level attention).

Stack: PyTorch (CNN / LSTM / BiLSTM / Attention) · FastAPI · React + Vite

## Run it (Windows PowerShell, from this folder)

```powershell
# 1. Python env (recreate it if you moved the folder)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

# 2. Smoke-test data + train the explainable model (about a minute on CPU)
python -m ml.make_sample_data
python -m ml.train --arch bilstm_attn

# 3. Backend (terminal 1)
Copy-Item .env.example .env
python -m uvicorn backend.app.main:app --reload --port 8000

# 4. Frontend (terminal 2)
npm install
npm run dev        # http://localhost:5173
```

Then open Claim Studio and press **Analyse claims**.

## Research experiments
```powershell
python -m ml.experiments                 # all suites -> reports/comparison.json
python -m ml.experiments --suites arch   # CNN vs LSTM vs BiLSTM vs Attention only
python -m ml.pca_embeddings              # PCA/SVD of embeddings -> reports/pca.json
python -m ml.demo.mcculloch_pitts        # neuron demo
python -m pytest                         # tests
```
Results show in the **Neural Lab** and **Research Journal** screens.

## Important: sample data vs real research
`data/sample` is synthetic and templated, so scores on it are near-perfect and prove nothing. For real results, train on FEVER: see `data/README.md`.

## Layout
```
backend/   FastAPI app: claim extraction, retrieval, verdict logic
ml/        models, training, experiments, FEVER converter, demos
frontend/  React screens (Archive, Claim Studio, Evidence Chamber, Context Graph, Stance Observatory, Neural Lab, Research Journal)
data/  docs/  reports/  tests/
```
See `docs/ARCHITECTURE.md` for the pipeline and syllabus map.
