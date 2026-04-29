# Research Questions

This repo studies representation learning for multivariate windows under changing operating modes. The goal is to learn useful feature spaces, not only classifiers.

## RQ1: Latent-Factor Recoverability

Can unsupervised embeddings recover latent factors and organize behavior in a controlled synthetic setting?

Primary factors:

- `mode_id`: full operating regime.
- `spectral_id`: temporal pattern family.
- `coupling_id`: cross-channel interaction level.
- `is_transition_window`: whether a window crosses a regime boundary.
- `mean_load`: continuous load or nuisance factor.

The RQ1 workflow is:

```text
generate controlled trajectories
-> split by trajectory
-> compute baselines
-> train or extract embeddings
-> evaluate probes, retrieval, clustering, and transition behavior
```

## RQ2: Method Comparison And Robustness

How do representation learning approaches differ in factor recovery, structure, robustness, and deployment cost?

Method families:

- reconstruction and masked reconstruction,
- contrastive methods such as TS2Vec-style encoders, TS-TCC, and TNC,
- hybrid losses combining contrastive and reconstruction terms,
- pretrained or foundation-model embeddings such as MOMENT,
- engineered-feature baselines.

Comparison dimensions:

- latent-factor recovery across clean/noisy datasets,
- local structure through retrieval,
- global structure through clustering,
- linear accessibility through probes,
- robustness to noise, device shifts, channel corruption, and OOD settings,
- practical cost: latency, memory footprint, CPU/GPU needs, and embedding size.

## RQ3: Evaluation Metric Reliability

To what extent do standard embedding metrics reflect true latent-factor recovery and downstream usefulness?

This question tracks when metrics disagree. Examples:

- high retrieval caused by temporal-neighbor bias,
- good probe accuracy but poor clustering,
- strong spectral recovery but weak coupling or load recovery,
- good clean performance but poor noisy or transition-window behavior,
- metrics that do not predict anomaly or downstream task performance.

The intended output is not a single winning score, but a clear diagnosis of what each representation has actually learned.
