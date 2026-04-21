# Table 3. Clustering Performance

Metrics are KMeans adjusted Rand index (ARI). MOMENT is mean +/- std over seeds 42/53/64.

| dataset | method | seeds | mode_ari | spectral_ari | coupling_ari | avg_factor_ari |
| --- | --- | --- | --- | --- | --- | --- |
| Clean long fair | MOMENT pretrained | 3 | 0.151 +/- 0.009 | 0.180 +/- 0.035 | 0.058 +/- 0.016 | 0.130 |
| Clean long fair | TS2Vec-style e80 | 1 | 0.153 | 0.207 | 0.006 | 0.122 |
| Clean long fair | TS2Vec-style tuned e120 | 1 | 0.157 | 0.194 | 0.004 | 0.118 |
| Noisy long fair | MOMENT pretrained | 3 | 0.143 +/- 0.008 | 0.210 +/- 0.046 | 0.041 +/- 0.025 | 0.132 |
| Noisy long fair | TS2Vec-style e80 | 1 | 0.132 | 0.208 | 0.020 | 0.120 |
| Noisy long fair | TS2Vec-style tuned e120 | 1 | 0.139 | 0.054 | 0.015 | 0.069 |
