# RQ1 Primary Probe Table

Linear-probe balanced accuracy for classification targets and R2 for `mean_load`. MOMENT is averaged over seeds 42/53/64.

| dataset | method | seeds | mode_bal_acc | spectral_bal_acc | coupling_bal_acc | transition_bal_acc | mean_load_r2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Clean long fair | Baseline FFT | 1 | 0.492 | 0.736 | 0.698 | 0.615 | 0.095 |
| Clean long fair | Baseline Summary | 1 | 0.497 | 0.703 | 0.661 | 0.533 | 0.238 |
| Clean long fair | Baseline Raw | 1 | 0.089 | 0.256 | 0.400 | 0.504 | -0.356 |
| Clean long fair | MOMENT pretrained | 3 | 0.440 +/- 0.005 | 0.712 +/- 0.006 | 0.581 +/- 0.031 | 0.607 +/- 0.039 | -0.929 +/- 0.303 |
| Clean long fair | TS2Vec-style e80 | 1 | 0.467 | 0.722 | 0.698 | 0.548 | 0.007 |
| Clean long fair | TS2Vec-style tuned e120 | 1 | 0.562 | 0.818 | 0.678 | 0.586 | 0.093 |
| Noisy long fair | Baseline FFT | 1 | 0.505 | 0.707 | 0.665 | 0.590 | 0.073 |
| Noisy long fair | Baseline Summary | 1 | 0.460 | 0.664 | 0.646 | 0.528 | 0.067 |
| Noisy long fair | Baseline Raw | 1 | 0.091 | 0.257 | 0.370 | 0.471 | -0.290 |
| Noisy long fair | MOMENT pretrained | 3 | 0.389 +/- 0.014 | 0.678 +/- 0.006 | 0.520 +/- 0.027 | 0.593 +/- 0.011 | -0.715 +/- 0.165 |
| Noisy long fair | TS2Vec-style e80 | 1 | 0.464 | 0.706 | 0.588 | 0.514 | -0.092 |
| Noisy long fair | TS2Vec-style tuned e120 | 1 | 0.455 | 0.743 | 0.604 | 0.516 | -0.272 |