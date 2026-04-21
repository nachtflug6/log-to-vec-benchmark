# Table 1. Latent Factor Recovery (RBF Probes)

Classification metrics are RBF-probe balanced accuracy. `mean_load_r2` is RBF-probe R2. MOMENT is mean +/- std over seeds 42/53/64.

| dataset | method | seeds | mode_bacc | spectral_bacc | coupling_bacc | transition_bacc | mean_load_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clean long fair | MOMENT pretrained | 3 | 0.466 +/- 0.012 | 0.756 +/- 0.018 | 0.633 +/- 0.042 | 0.576 +/- 0.042 | -0.037 +/- 0.049 |
| Clean long fair | TS2Vec-style e80 | 1 | 0.398 | 0.694 | 0.625 | 0.527 | -0.006 |
| Clean long fair | TS2Vec-style tuned e120 | 1 | 0.514 | 0.767 | 0.713 | 0.527 | -0.006 |
| Noisy long fair | MOMENT pretrained | 3 | 0.441 +/- 0.034 | 0.734 +/- 0.037 | 0.593 +/- 0.047 | 0.557 +/- 0.023 | -0.090 +/- 0.122 |
| Noisy long fair | TS2Vec-style e80 | 1 | 0.431 | 0.702 | 0.593 | 0.503 | -0.130 |
| Noisy long fair | TS2Vec-style tuned e120 | 1 | 0.450 | 0.733 | 0.637 | 0.494 | -0.151 |


# Table 2. Retrieval Performance

Metrics are precision@10 for same-label nearest-neighbor retrieval on factor labels. MOMENT is mean +/- std over seeds 42/53/64.

| dataset | method | seeds | mode_p@10 | spectral_p@10 | coupling_p@10 | avg_factor_p@10 |
| --- | --- | --- | --- | --- | --- | --- |
| Clean long fair | MOMENT pretrained | 3 | 0.340 +/- 0.017 | 0.592 +/- 0.011 | 0.526 +/- 0.024 | 0.486 |
| Clean long fair | TS2Vec-style e80 | 1 | 0.378 | 0.593 | 0.551 | 0.507 |
| Clean long fair | TS2Vec-style tuned e120 | 1 | 0.407 | 0.606 | 0.583 | 0.532 |
| Noisy long fair | MOMENT pretrained | 3 | 0.344 +/- 0.006 | 0.584 +/- 0.031 | 0.529 +/- 0.024 | 0.486 |
| Noisy long fair | TS2Vec-style e80 | 1 | 0.387 | 0.576 | 0.589 | 0.517 |
| Noisy long fair | TS2Vec-style tuned e120 | 1 | 0.385 | 0.567 | 0.578 | 0.510 |


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


# Table 4. Baseline vs Learned Embedding

Metrics are RBF-probe balanced accuracy for latent categorical factors and RBF-probe R2 for `mean_load`. `avg_factor_bacc` averages mode/spectral/coupling only.

| dataset | representation | type | seeds | mode_bacc | spectral_bacc | coupling_bacc | avg_factor_bacc | mean_load_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean long fair | Baseline FFT | hand-crafted baseline | 1 | 0.439 | 0.696 | 0.673 | 0.603 | -0.006 |
| Clean long fair | Baseline Summary | hand-crafted baseline | 1 | 0.431 | 0.653 | 0.634 | 0.573 | -0.006 |
| Clean long fair | Baseline Raw flatten | hand-crafted baseline | 1 | 0.327 | 0.551 | 0.622 | 0.500 | -0.006 |
| Clean long fair | MOMENT pretrained | learned embedding | 3 | 0.466 +/- 0.012 | 0.756 +/- 0.018 | 0.633 +/- 0.042 | 0.618 | -0.037 +/- 0.049 |
| Clean long fair | TS2Vec-style e80 | learned embedding | 1 | 0.398 | 0.694 | 0.625 | 0.572 | -0.006 |
| Clean long fair | TS2Vec-style tuned e120 | learned embedding | 1 | 0.514 | 0.767 | 0.713 | 0.665 | -0.006 |
| Noisy long fair | Baseline FFT | hand-crafted baseline | 1 | 0.432 | 0.681 | 0.614 | 0.576 | 0.028 |
| Noisy long fair | Baseline Summary | hand-crafted baseline | 1 | 0.428 | 0.641 | 0.669 | 0.579 | 0.097 |
| Noisy long fair | Baseline Raw flatten | hand-crafted baseline | 1 | 0.285 | 0.587 | 0.555 | 0.476 | 0.014 |
| Noisy long fair | MOMENT pretrained | learned embedding | 3 | 0.441 +/- 0.034 | 0.734 +/- 0.037 | 0.593 +/- 0.047 | 0.590 | -0.090 +/- 0.122 |
| Noisy long fair | TS2Vec-style e80 | learned embedding | 1 | 0.431 | 0.702 | 0.593 | 0.575 | -0.130 |
| Noisy long fair | TS2Vec-style tuned e120 | learned embedding | 1 | 0.450 | 0.733 | 0.637 | 0.607 | -0.151 |
