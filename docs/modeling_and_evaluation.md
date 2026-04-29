# Modeling And Evaluation Protocol

The main modeling target is:

```text
window X in R^(T x C) -> embedding z in R^d
```

The embedding should organize behavior in a way that supports several downstream checks.

## Modeling Families

Use strong baselines first:

- raw flattened windows,
- summary statistics,
- FFT or spectral summaries,
- cross-channel correlation and lag features,
- PCA or standardized feature projections.

Learned methods in this repo include:

- compact TCN/CNN encoders,
- TS2Vec-style contrastive encoders,
- masked or reconstruction-style objectives,
- hybrid contrastive plus reconstruction objectives,
- MOMENT pretrained embeddings.

MOMENT is useful as a pretrained reference. TS2Vec-style in-domain training is the default learned baseline when enough unlabeled windows are available.

## Positive Pair Policy

The training objective must match the evaluation goal.

Temporal-neighbor positives can create high retrieval scores without learning global factor structure. Conservative view-based positives, same-window crops, and factor-aware diagnostics are preferred for the formal benchmark.

Avoid augmentations that erase factors later evaluated as meaningful. Strong time warping, random permutation, or aggressive scaling can damage coupling, phase, and transition information.

## Evaluation Suite

Use multiple views:

- Linear and RBF probes for discrete factors.
- Ridge or SVR probes for continuous factors such as load.
- Retrieval metrics such as precision at k for local geometry.
- Clustering metrics such as ARI and NMI for global structure.
- Transition-window analysis reported separately from clean in-mode windows.
- Collapse diagnostics such as embedding variance and norm distribution.
- Multi-seed reporting for data generation and model initialization.

Linear probes are primary diagnostics because they reveal whether information is organized cleanly in the embedding. Nonlinear probes are secondary checks for entangled but present information.

## Known Failure Modes

Current context identified several recurring issues:

- training optimizes local temporal similarity while evaluation asks for factor recovery,
- retrieval can look strong because it matches the pretext task,
- spectral factors can dominate weaker factors such as load,
- transition windows can be blurred by smoothness objectives,
- baselines can outperform learned embeddings when simple features encode the target,
- single-seed results can be unstable.

Reports should explicitly discuss these failure modes rather than only listing aggregate metrics.
