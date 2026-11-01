"""PCA via SVD on learned word embeddings -> reports/pca.json.  python -m ml.pca_embeddings"""
import json
import sys

import numpy as np
import torch

path = sys.argv[1] if len(sys.argv) > 1 else "ml/checkpoints/bilstm_attn.pt"
ckpt = torch.load(path, map_location="cpu")
W = ckpt["state"]["embedding.weight"].numpy()[2:]
words = ckpt["vocab"][2:]
X = W - W.mean(0)
U, S, Vt = np.linalg.svd(X, full_matrices=False)
pts = X @ Vt[:2].T
var = (S**2 / (S**2).sum())[:2].tolist()
json.dump({"explained_variance": var, "points": [{"word": w, "x": float(x), "y": float(y)} for w, (x, y) in zip(words, pts)]},
          open("reports/pca.json", "w"), indent=1)
print(f"PCA written. Explained variance of 2 components: {sum(var):.1%}")
