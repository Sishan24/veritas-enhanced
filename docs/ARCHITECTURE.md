# VERITAS architecture

```
Article / statement
   -> claim extraction      backend/app/services/claims.py   (sentences, claim-worthiness, entities)
   -> evidence retrieval    backend/app/services/retriever.py (TF-IDF over the supplied passages)
   -> stance model          ml/models.py                      (MLP | CNN | LSTM | BiLSTM | BiLSTM+Attention)
   -> verdict aggregation   backend/app/services/verifier.py  (SUPPORTED | REFUTED | CONFLICTING | INSUFFICIENT)
   -> explainable report    frontend/ (evidence cards + attention highlights + constellation)
```

Research question: can a deep model learn the relationship between a claim and its evidence, while giving an inspectable justification?
It does not claim to determine absolute truth. No relevant, confident evidence means INSUFFICIENT, never false.

## Syllabus map
| Concept | Where |
| --- | --- |
| McCulloch-Pitts | `ml/demo/mcculloch_pitts.py` |
| MLP baseline, backprop | `ml/models.py` (`mlp`), `ml/train.py` |
| Activations | `--activation relu/sigmoid/tanh/gelu`, suite `activation` |
| SGD / Momentum / Nesterov | `--optimizer`, suite `optimizer` |
| PCA / SVD | `ml/pca_embeddings.py` |
| CNN, LSTM, BiLSTM, Attention | `ml/models.py` |
| Dropout, weight decay, early stopping | `--dropout`, `--weight-decay`, `--patience`, suite `regularization` |

## Metrics
Accuracy, macro-F1, precision, recall, confusion matrix, training time, inference latency (all in `reports/<run>/metrics.json`).
Retrieval Precision@K / Recall@K still needs a gold-evidence set (FEVER provides one); add it in `ml/` when you move to real data.
