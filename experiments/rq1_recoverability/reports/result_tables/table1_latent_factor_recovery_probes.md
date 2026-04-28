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
