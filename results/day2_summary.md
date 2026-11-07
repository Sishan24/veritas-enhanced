# VERITAS Day 2 Evaluation Summary

## 1. Main Results: Pair vs Passage (Pooled)

| arch | hidden | n_params | split | n | accuracy mean ± std | macro-F1 mean ± std |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | 128 | 239,379 | pair | 2 | 0.9889 ± 0.0047 | 0.9889 ± 0.0049 |
| cnn | 128 | 265,792 | pair | 2 | 0.9983 ± 0.0008 | 0.9983 ± 0.0008 |
| lstm | 128 | 344,211 | pair | 2 | 0.9717 ± 0.0228 | 0.9716 ± 0.0227 |
| bilstm | 128 | 527,507 | pair | 2 | 0.9900 ± 0.0016 | 0.9903 ± 0.0015 |
| bilstm_attn | 128 | 593,043 | pair | 2 | 0.9928 ± 0.0024 | 0.9930 ± 0.0023 |
| lstm | 256 | 790,163 | pair | 2 | 0.9889 ± 0.0047 | 0.9890 ± 0.0048 |
| majority | 0 | 0 | pair | 1 | 0.3422 ± n/a | 0.1700 ± n/a |
| tfidf_logreg | 0 | 0 | pair | 1 | 0.6800 ± n/a | 0.6886 ± n/a |
| mlp | 128 | 206,146 | passage (pooled) | 6 | 0.5680 ± 0.0590 | 0.5426 ± 0.0713 |
| cnn | 128 | 232,559 | passage (pooled) | 6 | 0.5946 ± 0.0855 | 0.5847 ± 0.0905 |
| lstm | 128 | 310,978 | passage (pooled) | 6 | 0.6239 ± 0.0505 | 0.6208 ± 0.0467 |
| bilstm | 128 | 494,274 | passage (pooled) | 6 | 0.5285 ± 0.0406 | 0.5332 ± 0.0382 |
| bilstm_attn | 128 | 559,810 | passage (pooled) | 6 | 0.5315 ± 0.0575 | 0.5336 ± 0.0546 |
| lstm | 256 | 756,930 | passage (pooled) | 6 | 0.6278 ± 0.0401 | 0.6259 ± 0.0343 |
| majority | 0 | 0 | passage (pooled) | 3 | 0.3385 ± 0.0178 | 0.1685 ± 0.0066 |
| tfidf_logreg | 0 | 0 | passage (pooled) | 3 | 0.4270 ± 0.0694 | 0.4029 ± 0.0724 |

*Note: `majority` and `tfidf_logreg` are deterministic baselines; duplicate model seeds (ms0 and ms1) yield identical results and are not counted as independent samples. For individual splits, n=1 (sample std is n/a). For the pooled passage view, metrics reflect the spread across the 3 split seeds only (n=3).*

## 2. Passage Split Breakdown (Split Seed Effect vs Model Seed)

| arch | hidden | n_params | split | n | accuracy mean ± std | macro-F1 mean ± std |
| --- | --- | --- | --- | --- | --- | --- |
| mlp | 128 | 206,079 | passage_ss0 | 2 | 0.5967 ± 0.0613 | 0.5922 ± 0.0682 |
| mlp | 128 | 205,479 | passage_ss1 | 2 | 0.5594 ± 0.0306 | 0.5072 ± 0.0629 |
| mlp | 128 | 206,879 | passage_ss2 | 2 | 0.5478 ± 0.1006 | 0.5285 ± 0.0948 |
| cnn | 128 | 232,492 | passage_ss0 | 2 | 0.5050 ± 0.0244 | 0.4996 ± 0.0257 |
| cnn | 128 | 231,892 | passage_ss1 | 2 | 0.6844 ± 0.0503 | 0.6822 ± 0.0532 |
| cnn | 128 | 233,292 | passage_ss2 | 2 | 0.5944 ± 0.0346 | 0.5724 ± 0.0602 |
| lstm | 128 | 310,911 | passage_ss0 | 2 | 0.5950 ± 0.0071 | 0.5972 ± 0.0032 |
| lstm | 128 | 310,311 | passage_ss1 | 2 | 0.6594 ± 0.0401 | 0.6591 ± 0.0486 |
| lstm | 128 | 311,711 | passage_ss2 | 2 | 0.6172 ± 0.0825 | 0.6062 ± 0.0638 |
| bilstm | 128 | 494,207 | passage_ss0 | 2 | 0.5706 ± 0.0134 | 0.5750 ± 0.0104 |
| bilstm | 128 | 493,607 | passage_ss1 | 2 | 0.4856 ± 0.0267 | 0.4913 ± 0.0086 |
| bilstm | 128 | 495,007 | passage_ss2 | 2 | 0.5294 ± 0.0102 | 0.5332 ± 0.0110 |
| bilstm_attn | 128 | 559,743 | passage_ss0 | 2 | 0.4928 ± 0.0165 | 0.4958 ± 0.0338 |
| bilstm_attn | 128 | 559,143 | passage_ss1 | 2 | 0.5711 ± 0.0126 | 0.5695 ± 0.0266 |
| bilstm_attn | 128 | 560,543 | passage_ss2 | 2 | 0.5306 ± 0.0998 | 0.5356 ± 0.0871 |
| lstm | 256 | 756,863 | passage_ss0 | 2 | 0.5967 ± 0.0629 | 0.5959 ± 0.0507 |
| lstm | 256 | 756,263 | passage_ss1 | 2 | 0.6289 ± 0.0000 | 0.6303 ± 0.0123 |
| lstm | 256 | 757,663 | passage_ss2 | 2 | 0.6578 ± 0.0189 | 0.6515 ± 0.0052 |
| majority | 0 | 0 | passage_ss0 | 1 | 0.3400 ± n/a | 0.1692 ± n/a |
| majority | 0 | 0 | passage_ss1 | 1 | 0.3556 ± n/a | 0.1749 ± n/a |
| majority | 0 | 0 | passage_ss2 | 1 | 0.3200 ± n/a | 0.1616 ± n/a |
| tfidf_logreg | 0 | 0 | passage_ss0 | 1 | 0.3533 ± n/a | 0.3319 ± n/a |
| tfidf_logreg | 0 | 0 | passage_ss1 | 1 | 0.4367 ± n/a | 0.4002 ± n/a |
| tfidf_logreg | 0 | 0 | passage_ss2 | 1 | 0.4911 ± n/a | 0.4765 ± n/a |

*Note: Each split seed row for neural architectures averages n=2 model seeds (ms0 and ms1). Baselines have n=1 per split seed.*

## 3. Capacity Control: LSTM (h=256) vs BiLSTM (h=128)

| Architecture | Hidden | Split | n_params | Ratio (vs BiLSTM) | Accuracy (mean ± std) | Macro-F1 (mean ± std) |
| --- | --- | --- | --- | --- | --- | --- |
| bilstm | 128 | pair | 527,507 | 1.00x | 0.9900 ± 0.0016 | 0.9903 ± 0.0015 |
| lstm | 256 | pair | 790,163 | 1.50x | 0.9889 ± 0.0047 | 0.9890 ± 0.0048 |
| bilstm | 128 | passage (pooled) | 494,274 | 1.00x | 0.5285 ± 0.0406 | 0.5332 ± 0.0382 |
| lstm | 256 | passage (pooled) | 756,930 | 1.53x | 0.6278 ± 0.0401 | 0.6259 ± 0.0343 |

## 4. Findings and Analysis

1. In the pooled passage evaluation, the macro-F1 differences between lstm_h128 (0.6208 ± 0.0467), lstm_h256 (0.6259 ± 0.0343), and cnn_h128 (0.5847 ± 0.0905) fall within their overlapping standard deviation intervals and are therefore not statistically distinguishable.
2. Similarly, bilstm_h128 (0.5332 ± 0.0382) and bilstm_attn_h128 (0.5336 ± 0.0546) exhibit virtually identical mean macro-F1 with overlapping standard deviations, showing no distinguishable difference between bidirectional models with and without attention.
3. With only 2 model seeds per split configuration, the reported sample standard deviation is a rough estimate, and an observed difference smaller than this spread does not constitute evidence that one architecture outperforms another.
4. Comparing lstm_h256 (756,930 parameters in passage) with bilstm_h128 (494,274 parameters) yields an actual parameter ratio of 1.53x, meaning the models have substantially different capacities rather than matched parameter counts.
5. In passage splits, bilstm_h128 does not hold an advantage over lstm_h256 or lstm_h128; lstm_h256 attains 0.6259 ± 0.0343 macro-F1 while bilstm_h128 attains 0.5332 ± 0.0382, indicating that bidirectional modeling does not improve performance over unidirectional LSTM on passage partitions.
6. Across all neural architectures, moving from pair splits to passage splits causes macro-F1 to drop by 36 to 46 percentage points (from 0.9716–0.9983 on pair down to 0.5332–0.6259 on pooled passage), confirming severe performance inflation from evidence leakage in pair mode.
7. Partitioning variance across the three passage split seeds (e.g. cnn_h128 macro-F1 ranges from 0.4996 on ss0 to 0.6822 on ss1) produces a much larger spread than random model initialization seeds within any given split (which typically differ by 0.01 to 0.06).
8. All 64 scheduled runs completed successfully without failures, but split seed 1 exhibits an unusual divergence where cnn_h128 macro-F1 jumps to 0.6822 while bilstm_h128 drops to 0.4913, reflecting marked sensitivity to the specific passage split.
