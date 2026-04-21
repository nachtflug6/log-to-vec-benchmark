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
