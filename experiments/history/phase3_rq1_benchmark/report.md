# Phase 3 Report: Formal RQ1 Benchmark

Branch: `experiment1` (integrated into `main`)
Status: **Completed — baseline established, problems identified**.
Code: [experiments/rq1_recoverability/](../../../experiments/rq1_recoverability/)
Artifacts: [experiments/rq1_recoverability/artifacts/](../../../experiments/rq1_recoverability/artifacts/)
Reports: [experiments/rq1_recoverability/reports/](../../../experiments/rq1_recoverability/reports/)

---

## Objective

Establish a rigorous, reproducible benchmark for the three research questions:
- **RQ1**: Can unsupervised embeddings recover latent factors in a controlled synthetic setting?
- **RQ2**: How do method families compare across factor recovery, robustness, and cost?
- **RQ3**: When do evaluation metrics disagree or mislead?

---

## What Was Tried

### The FRS Generator

File: `experiments/rq1_recoverability/src/rq1/generation/factorized_regime_sequence_generator.py`

Factorised Regime Sequence (FRS) generator. Trajectories are composed of regimes drawn from
a product space of latent factors:

| Factor         | Type       | Values / Range |
|----------------|------------|----------------|
| spectral_id    | discrete   | 4 families: periodic, damped, multi-periodic, quasi-aperiodic |
| coupling_id    | discrete   | 3 levels: low, medium, high cross-channel interaction |
| mode_id        | discrete   | 12 combinations (spectral × coupling) |
| mean_load      | continuous | varies per regime, affects dynamics and noise |
| is_transition  | boolean    | window crosses a regime boundary |

Device effects (gain, bias, dropout) and process/observation noise are added on top.

Two canonical profiles:
- `frs_clean_vnext_long`: minimal noise, intended to expose factor structure cleanly.
- `frs_noisy_vnext_long`: process noise, observation noise, device shifts, channel dropout.

### Datasets and Splits

Generator script: `experiments/rq1_recoverability/scripts/01_generate_frs_dataset.py`
Split script: `experiments/rq1_recoverability/scripts/02_create_dataset_splits.py`

- **Splitting**: trajectory-level (not window-level). 70/15/15 train/val/test.
  Every split includes a leakage report.
- Dataset configs: `experiments/rq1_recoverability/configs/datasets/frs_clean_vnext_long.yaml`
  and `frs_noisy_vnext_long.yaml`.
- Dataset registry: `experiments/rq1_recoverability/manifests/datasets.json`.

### Baselines

Script: `experiments/rq1_recoverability/scripts/03_compute_baseline_representations.py`
Config: `experiments/rq1_recoverability/configs/representations/`

Three hand-crafted baselines:
- **Baseline FFT**: power spectrum features per channel (strongest for spectral factor).
- **Baseline Summary**: mean, std, min, max, percentiles per channel.
- **Baseline Raw flatten**: raw window values flattened.

### Methods Evaluated

| Method | Script | Seeds |
|--------|--------|-------|
| TS2Vec-style contrastive e80 | `11_run_ts2vec_recoverability.py` | 1 (seed 42) |
| TS2Vec-style contrastive e120 (tuned) | `11_run_ts2vec_recoverability.py` | 1 (seed 42) |
| MOMENT pretrained embeddings | `07_run_frs_moment_recoverability.py` | 3 (42, 53, 64) |
| Masked reconstruction | `12_run_masked_reconstruction_recoverability.py` | — |
| Masked multiscale reconstruction | `13_run_masked_multiscale_reconstruction_recoverability.py` | — |

TS2Vec-style encoder: TCN backbone trained with hierarchical contrastive loss (TS2Vec
formulation). Positive pairs defined by temporal subseries overlap.

MOMENT: foundation model for time-series (pretrained, no fine-tuning). Embeddings extracted
directly from the pretrained model. Script: `05_export_moment_pretrained_embeddings.py`.

### Evaluation Protocol

Script: `experiments/rq1_recoverability/scripts/06_evaluate_recoverability.py`
Config: `experiments/rq1_recoverability/configs/evaluation/rq1_core.yaml`

Per factor: linear probe, RBF probe, retrieval precision@k, clustering ARI/NMI.
Evaluation dimensions:
- **Primary**: linear probe balanced accuracy, retrieval precision@10.
- **Secondary**: clustering ARI, transition analysis, robustness.
- **Target factors**: mode_id, spectral_id, coupling_id, is_transition_window, mean_load.

All evaluation is on the held-out test trajectories.

---

## Results in Full

### Table 1: RBF Probe Balanced Accuracy

Source: [reports/result_tables/table1_latent_factor_recovery_probes.md](../../../experiments/rq1_recoverability/reports/result_tables/table1_latent_factor_recovery_probes.md)

| Dataset   | Method          | Seeds | mode     | spectral  | coupling  | transition | load R²    |
|-----------|-----------------|-------|----------|-----------|-----------|------------|------------|
| Clean     | MOMENT          | 3     | 0.466 ±0.012 | 0.756 ±0.018 | 0.633 ±0.042 | 0.576 ±0.042 | −0.037 ±0.049 |
| Clean     | TS2Vec e80      | 1     | 0.398    | 0.694     | 0.625     | 0.527      | −0.006     |
| Clean     | TS2Vec e120     | 1     | **0.514** | **0.767** | **0.713** | 0.527     | −0.006     |
| Noisy     | MOMENT          | 3     | 0.441 ±0.034 | 0.734 ±0.037 | 0.593 ±0.047 | 0.557 ±0.023 | −0.090 ±0.122 |
| Noisy     | TS2Vec e80      | 1     | 0.431    | 0.702     | 0.593     | 0.503      | −0.130     |
| Noisy     | TS2Vec e120     | 1     | 0.450    | 0.733     | 0.637     | 0.494      | **−0.151** |

Chance level: mode 1/12 ≈ 0.083, spectral 1/4 = 0.25, coupling 1/3 = 0.33.
All methods substantially above chance for spectral and coupling, but marginal for mode.
Load R² ≤ 0 for every method: load is completely unlearned.

### Table 2: Retrieval Precision@10

Source: [reports/result_tables/table2_retrieval_performance.md](../../../experiments/rq1_recoverability/reports/result_tables/table2_retrieval_performance.md)

| Dataset   | Method          | mode  | spectral | coupling | avg   |
|-----------|-----------------|-------|----------|----------|-------|
| Clean     | MOMENT          | 0.340 | 0.592    | 0.526    | 0.486 |
| Clean     | TS2Vec e80      | 0.378 | 0.593    | 0.551    | 0.507 |
| Clean     | TS2Vec e120     | **0.407** | **0.606** | **0.583** | **0.532** |
| Noisy     | MOMENT          | 0.344 | 0.584    | 0.529    | 0.486 |
| Noisy     | TS2Vec e80      | 0.387 | 0.576    | 0.589    | 0.517 |
| Noisy     | TS2Vec e120     | 0.385 | 0.567    | 0.578    | 0.510 |

Retrieval is moderate (0.3–0.6), not the artificially inflated values seen in Phase 2.
The FRS dataset has more mode diversity (12 modes) which makes retrieval harder.

Note: retrieval precision@10 on the FRS benchmark is more honest than Phase 2 R@5 (which was
0.96–0.99) because the FRS test set has better trajectory separation.

### Table 3: Clustering ARI

Source: [reports/result_tables/table3_clustering_performance.md](../../../experiments/rq1_recoverability/reports/result_tables/table3_clustering_performance.md)

| Dataset   | Method          | mode  | spectral | coupling | avg   |
|-----------|-----------------|-------|----------|----------|-------|
| Clean     | MOMENT          | 0.151 | 0.180    | 0.058    | 0.130 |
| Clean     | TS2Vec e80      | 0.153 | **0.207** | 0.006   | 0.122 |
| Clean     | TS2Vec e120     | **0.157** | 0.194 | 0.004   | 0.118 |
| Noisy     | MOMENT          | 0.143 | 0.210    | 0.041    | 0.132 |
| Noisy     | TS2Vec e80      | 0.132 | 0.208    | 0.020    | 0.120 |
| Noisy     | TS2Vec e120     | 0.139 | 0.054    | 0.015    | 0.069 |

Clustering ARI is universally low across all methods. Mode ARI (0.15) is particularly low
given 12 classes. Coupling ARI is near zero for TS2Vec despite coupling probe of 0.71.
More training (e80→e120) does not improve clustering and degrades it on noisy data.
MOMENT is marginally more stable than TS2Vec under noise for spectral clustering.

### Table 4: Baseline vs Learned (avg factor balanced accuracy)

Source: [reports/result_tables/table4_baseline_vs_learned_embedding.md](../../../experiments/rq1_recoverability/reports/result_tables/table4_baseline_vs_learned_embedding.md)

| Method           | Type     | Clean avg | Noisy avg |
|------------------|----------|-----------|-----------|
| Baseline FFT     | baseline | 0.603     | 0.576     |
| Baseline Summary | baseline | 0.573     | 0.579     |
| Baseline Raw     | baseline | 0.500     | 0.476     |
| MOMENT           | learned  | 0.618     | 0.590     |
| TS2Vec e80       | learned  | 0.572     | 0.575     |
| TS2Vec e120      | learned  | **0.665** | **0.607** |

Gap between best learned (TS2Vec e120) and best baseline (FFT): **+6.2pp clean, +3.1pp noisy**.
At this margin, TS2Vec e120 can claim improvement over FFT, but it is not dramatic.
TS2Vec e80 is actually *below* FFT. MOMENT narrowly beats FFT.

---

## Key Findings

### Finding 1: Load factor is never learned

Load R² ≤ 0 in every method, every setting (clean and noisy). The factor is present in the
data but no method extracts it. This is not a baseline limitation—FFT also fails (R²=−0.006).
Load variation is subtle (affects amplitude and noise) and has no spectral signature; the
training objectives have no incentive to preserve it.

**Implication**: An explicit regression objective targeting load is needed. Current
unsupervised methods cannot recover it.

### Finding 2: Probe accuracy and clustering ARI dissociate

TS2Vec e120 achieves coupling probe 0.713 (clean) but coupling clustering ARI 0.004. This
means the embedding encodes coupling information in a linearly accessible way but does not
form globally separated coupling clusters. Local structure (probe) is present; global structure
(clustering) is absent.

**Implication**: The loss enforces local consistency but not global structure. Probe accuracy
without clustering ARI gives an overly optimistic picture.

### Finding 3: More training helps probes but not clustering

TS2Vec e80 → e120 improves all probe scores (mode: 0.398→0.514, spectral: 0.694→0.767,
coupling: 0.625→0.713). However, clustering ARI does not improve and actually degrades for
coupling (0.006→0.004 clean, 0.020→0.015 noisy).

**Implication**: The training objective is optimising linear accessibility of factors without
producing well-separated geometric structure. Training longer is not the solution.

### Finding 4: Noise degrades all methods, MOMENT most robust

Noisy vs clean delta for avg factor balanced accuracy:
- Baseline FFT: −0.027
- MOMENT: −0.028
- TS2Vec e120: **−0.058** (nearly 2× worse than others)

MOMENT is more robust under noise than the trained-from-scratch methods.

**Implication**: Pretrained foundation models generalise better to noise. From-scratch
training on noisy data requires specific robustness strategies.

### Finding 5: Multi-seed variance (MOMENT only)

Over seeds 42/53/64, MOMENT standard deviations:
- Clean: mode ±0.012, spectral ±0.018, coupling ±0.042
- Noisy: mode ±0.034, spectral ±0.037, coupling ±0.047

Coupling shows the highest variance (±0.042 clean), confirming it is the noisiest signal.
TS2Vec has only 1 seed, so its stability is unknown.

**Implication**: All TS2Vec results should be treated with caution. Multi-seed reporting
(3+ seeds) should be mandatory for any future method comparison.

### Finding 6: Temporal-neighbour bias is reduced but not eliminated on FRS

Phase 2 showed R@5 of 0.96–0.99. Phase 3 retrieval precision@10 is 0.34–0.61, which is
more honest. The FRS dataset's trajectory-level splits and larger mode space reduce the
bias, but retrieval is still not a reliable proxy for semantic understanding.

---

## What the RQ1 Baseline Tells Us

The RQ1 benchmark establishes the following baseline:

| Claim | Evidence |
|-------|----------|
| Any method must exceed Baseline FFT avg 60.3% (clean) to be worth reporting | Table 4 |
| Load recovery requires explicit objective; current methods score ≤ 0 | Tables 1, 4 |
| Mode clustering requires global-structure objectives; current ARI ≤ 0.157 | Table 3 |
| TS2Vec e120 is the strongest from-scratch baseline; future methods should beat it | Tables 1–4 |
| MOMENT is the strongest pretrained reference; future pretrained methods should beat it | Tables 1–4 |

---

## Infrastructure Established

This phase leaves working, tested infrastructure:

- FRS generator with clean/noisy/OOD profiles and reproducible seeds.
- Trajectory-level split creation with leakage checking.
- Numbered pipeline scripts 01–13 for the full RQ1 workflow.
- Experiment registry schema and run metadata template.
- Alvis HPC integration (Slurm + Apptainer), smoke-tested 2026-04-29.
- Multi-view evaluation: probes + retrieval + clustering + transition + robustness.

This infrastructure should not be discarded or rebuilt. Next experiments plug into it.

---

## What NOT to Repeat

- Running TS2Vec or MOMENT again without changes and calling it a new result.
- Reporting only probe accuracy without clustering ARI.
- Single-seed experiments for any new method.
- Using frs_clean_v3 or frs_clean (older profiles); use `frs_*_vnext_long` variants only.
- Claiming load is recovered by any method without showing R² > 0.1.
