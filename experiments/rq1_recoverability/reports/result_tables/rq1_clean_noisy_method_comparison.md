# RQ1 Clean vs Noisy Method Comparison

Metrics: classification columns are linear-probe balanced accuracy; `mean_load_r2` is linear-probe R2. Retrieval columns are precision@1. Clustering columns are KMeans ARI. MOMENT values are mean +/- population std over seeds 42/53/64; TS2Vec and baselines are single runs.

| dataset | method | seeds | mode_bal_acc | spectral_bal_acc | coupling_bal_acc | transition_bal_acc | mean_load_r2 | mode_p@1 | spectral_p@1 | coupling_p@1 | transition_p@1 | mode_ari | spectral_ari | coupling_ari | transition_ari |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean long fair | Baseline FFT | 1 | 0.492 | 0.736 | 0.698 | 0.615 | 0.095 |  |  |  |  |  |  |  |  |
| Clean long fair | Baseline Summary | 1 | 0.497 | 0.703 | 0.661 | 0.533 | 0.238 |  |  |  |  |  |  |  |  |
| Clean long fair | Baseline Raw | 1 | 0.089 | 0.256 | 0.400 | 0.504 | -0.356 |  |  |  |  |  |  |  |  |
| Clean long fair | MOMENT pretrained | 3 | 0.440 +/- 0.005 | 0.712 +/- 0.006 | 0.581 +/- 0.031 | 0.607 +/- 0.039 | -0.929 +/- 0.303 | 0.495 +/- 0.035 | 0.693 +/- 0.010 | 0.640 +/- 0.032 | 0.686 +/- 0.027 | 0.144 +/- 0.004 | 0.182 +/- 0.029 | 0.058 +/- 0.013 | -0.001 +/- 0.004 |
| Clean long fair | TS2Vec-style e80 | 1 | 0.467 | 0.722 | 0.698 | 0.548 | 0.007 | 0.604 | 0.761 | 0.729 | 0.705 | 0.153 | 0.207 | 0.006 | -0.004 |
| Clean long fair | TS2Vec-style tuned e120 | 1 | 0.562 | 0.818 | 0.678 | 0.586 | 0.093 | 0.705 | 0.780 | 0.819 | 0.717 | 0.157 | 0.194 | 0.004 | -0.005 |
| Noisy long fair | Baseline FFT | 1 | 0.505 | 0.707 | 0.665 | 0.590 | 0.073 |  |  |  |  |  |  |  |  |
| Noisy long fair | Baseline Summary | 1 | 0.460 | 0.664 | 0.646 | 0.528 | 0.067 |  |  |  |  |  |  |  |  |
| Noisy long fair | Baseline Raw | 1 | 0.091 | 0.257 | 0.370 | 0.471 | -0.290 |  |  |  |  |  |  |  |  |
| Noisy long fair | MOMENT pretrained | 3 | 0.389 +/- 0.014 | 0.678 +/- 0.006 | 0.520 +/- 0.027 | 0.593 +/- 0.011 | -0.715 +/- 0.165 | 0.553 +/- 0.003 | 0.709 +/- 0.007 | 0.676 +/- 0.012 | 0.709 +/- 0.050 | 0.147 +/- 0.007 | 0.208 +/- 0.036 | 0.043 +/- 0.021 | 0.022 +/- 0.012 |
| Noisy long fair | TS2Vec-style e80 | 1 | 0.464 | 0.706 | 0.588 | 0.514 | -0.092 | 0.700 | 0.773 | 0.778 | 0.744 | 0.132 | 0.208 | 0.020 | 0.003 |
| Noisy long fair | TS2Vec-style tuned e120 | 1 | 0.455 | 0.743 | 0.604 | 0.516 | -0.272 | 0.705 | 0.771 | 0.773 | 0.780 | 0.139 | 0.054 | 0.015 | -0.001 |