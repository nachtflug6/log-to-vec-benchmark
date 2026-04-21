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

| Factor        | Clean (Baseline) | Clean (MOMENT) | Clean (TS2Vec e80) | Clean (TS2Vec e120) | Noisy (Baseline) | Noisy (MOMENT) | Noisy (TS2Vec e80) | Noisy (TS2Vec e120) |
|---------------|------------------|----------------|--------------------|---------------------|------------------|----------------|--------------------|---------------------|
| Mode          | 0.439            | 0.466          | 0.398              | 0.514               | 0.432            | 0.441          | 0.431              | 0.450               |
| Spectral      | 0.696            | 0.756          | 0.694              | 0.767               | 0.681            | 0.734          | 0.702              | 0.733               |
| Coupling      | 0.673            | 0.633          | 0.625              | 0.713               | 0.614            | 0.593          | 0.593              | 0.637               |
| Avg Factor    | 0.603            | 0.618          | 0.572              | 0.665               | 0.576            | 0.590          | 0.575              | 0.607               |
| Load (R²)     | -0.006           | -0.037         | -0.006             | -0.006              | 0.028            | -0.090         | -0.130             | -0.151              |