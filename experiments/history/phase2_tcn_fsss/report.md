# Phase 2 Report: TCN Hybrid on FSSS

Branch: `version2`
Status: **Completed with negative result**. Code in [src/version2/](../../../src/version2/).
Artifacts in [outputs/version2/](../../../outputs/version2/).

---

## Objective

Move beyond toy logs to a factorised synthetic dataset with known ground truth. Train a
Temporal Convolutional Network (TCN) with a hybrid contrastive + reconstruction loss and
evaluate whether the learned embedding recovers latent factors. Compare against hand-crafted
feature baselines.

---

## What Was Tried

### Architecture

**TCN Hybrid Encoder** (`src/version2/models/tcn_hybrid.py`):
- Dilated residual Conv1d blocks (exponential dilation: 1, 2, 4, … → broad temporal receptive field)
- Three heads from a shared backbone:
  - **Embedding head**: downstream representation
  - **Projection head**: contrastive loss input (MLP → unit sphere)
  - **Reconstruction head**: masked reconstruction target

**Hybrid loss** (`src/version2/training/hybrid_losses.py`):
- Contrastive term: NT-Xent between augmented views of the same window
- Reconstruction term: MSE over masked spans
- Loss = α × contrastive + (1−α) × reconstruction

### Data: FSSS Datasets

Simpler, earlier variants of the FRS generator. Factorised State-Space Sequence format.
Windows extracted from trajectories.

Key datasets used:
- `outputs/easy_clean/training_history.json` — clean FSSS training
- `outputs/easy_noise/training_history.json` — noisy FSSS training
- Earlier FSSS splits: not trajectory-level; had window-level overlap issues in some variants

Factors available: mode_id, spectral_id, coupling_id, is_transition_window, mean_load.

### Training

Script: `examples/fsss/train_tcn_hybrid.py`
- Augmentation: Gaussian noise, time shifts
- Span masking for reconstruction objective
- Checkpointing: best model saved by validation loss
- Configs: `experiments/rq1_recoverability/configs/models/tcn_encoder.yaml`

### Baselines

`src/version2/evaluation/baseline_features.py`:
- **FFT**: power spectrum features per channel
- **Summary**: mean, std, min, max, percentiles per channel
- **Raw flatten**: raw window values flattened

### Evaluation Suite

`src/version2/evaluation/eval_v2.py` (also used in `examples/fsss/run_eval_v2.py`):
- Linear and RBF probe classifiers/regressors per factor
- Retrieval: precision@k, recall@k, MAP, NDCG
- Clustering: k-means ARI, NMI, purity, silhouette
- Transition analysis: separate probe accuracy for clean vs transition windows

---

## Results

### Primary comparison (from outputs/version2/reports/compare/baseline_vs_ours.md)

Probe balanced accuracy on FSSS test set. "Ours" = TCN hybrid learned embedding.

| Factor          | Best Baseline | Best Baseline Score | Learned Best | Delta  |
|-----------------|---------------|---------------------|--------------|--------|
| mode_id         | Summary/linear | 0.277              | RBF: 0.267   | **−0.010 (baseline wins)** |
| spectral_id     | Summary/RBF   | 0.701               | Linear: 0.653 | **−0.048 (baseline wins)** |
| coupling_id     | Summary/linear | 0.413              | Linear: 0.423 | +0.010 (marginal) |
| is_transition   | FFT/linear    | 0.558               | RBF: 0.534   | **−0.024 (baseline wins)** |
| mean_load (R²)  | All near zero | ≈0                  | ≈0           | tied |

**The learned TCN embedding does not outperform hand-crafted baselines on any factor except
coupling by a negligible margin.**

### Detailed probe table (from outputs/version2/reports/run_003/full_report.md)

| Factor               | Probe  | Accuracy | Balanced Acc |
|----------------------|--------|----------|-------------|
| mode_id              | linear | 0.244    | 0.266       |
| mode_id              | RBF    | 0.242    | 0.267       |
| spectral_id          | linear | 0.657    | 0.653       |
| spectral_id          | RBF    | 0.643    | 0.641       |
| coupling_id          | linear | 0.411    | 0.423       |
| coupling_id          | RBF    | 0.399    | 0.408       |
| is_transition_window | linear | 0.597    | 0.526       |
| is_transition_window | RBF    | 0.635    | 0.534       |
| mean_load            | linear | —        | —  (R²: −0.136) |
| mean_load            | RBF    | —        | —  (R²: 0.004) |

### Retrieval (from outputs/version2/reports/run_003/full_report.md)

| Factor               | R@1   | R@5   | R@10  |
|----------------------|-------|-------|-------|
| mode_id              | 0.780 | 0.961 | 0.978 |
| spectral_id          | 0.862 | 0.983 | 0.990 |
| coupling_id          | 0.850 | 0.993 | 1.000 |
| is_transition_window | 0.698 | 0.969 | 0.995 |

R@5 is extremely high. This is the temporal-neighbour bias: the training objective directly
optimises neighbour similarity, and retrieval measures nearest neighbours. High retrieval here
does not indicate semantic understanding.

### Clustering (from outputs/version2/reports/run_003/full_report.md)

| Factor               | ARI    | NMI    | Purity  |
|----------------------|--------|--------|---------|
| mode_id              | 0.134  | 0.339  | 0.345   |
| spectral_id          | 0.030  | 0.079  | 0.350   |
| coupling_id          | 0.020  | 0.021  | 0.447   |
| is_transition_window | −0.008 | 0.001  | 0.640   |

ARI is very low across all factors. The model does not form globally separable clusters.
Mode ARI (0.134) is the highest, but still near-random for 12 classes.

### Transition analysis (from full_report.md)

| Metric            | Clean  | Transition |
|-------------------|--------|------------|
| Mode accuracy     | 0.282  | 0.223      |
| Retrieval@5       | 1.000  | 0.940      |

Transition windows are harder: mode accuracy drops from 28% to 22%, retrieval from 100% to 94%.
The model cannot correctly classify mode during transitions.

### Transition difficulty by distance from boundary

| Distance from boundary | Samples | Mode Accuracy | Balanced Acc |
|------------------------|---------|---------------|--------------|
| Near (≤ threshold)     | 54      | 0.296         | 0.305        |
| Mid                    | 89      | 0.258         | 0.280        |
| Far                    | 122     | 0.164         | 0.188        |

**Counter-intuitively**, windows far from the boundary are harder to classify than near ones.
This likely reflects temporal smoothing: the model's representation blurs across the trajectory,
making stable mid-segment windows less discriminable than transitions which have both old and new
regime signals present.

---

## Observations and Failure Analysis

### Why probes are low despite high retrieval

The training objective (temporal-neighbour contrastive) directly rewards pulling together
windows that are temporally close. Retrieval measures how often nearest neighbours share the
same label—which they do if temporal proximity correlates with mode identity (it does, within
a stable regime). But the model never learns to pull together windows from *different
trajectories* in the same mode, which is what probes and clustering measure.

### Why baselines win on spectral

FFT features directly encode frequency content, which is the dominant signal in the FSSS
spectral-family factor. The TCN sees the same signal but does not extract frequency information
more efficiently than FFT when trained with temporal-neighbour loss.

### Why load is never learned

Load affects signal amplitude and noise level slightly, but these effects are subtle. The
training loss (neighbour similarity + reconstruction) has no incentive to preserve load
information. The projection head discards any useful load signal in favour of reconstruction
and temporal-smoothness objectives.

### The eval_v2 degenerate result (DO NOT CITE)

An earlier evaluation run (outputs/eval_v2/) showed mode probe accuracy 94.7% and load R²=1.0.
These numbers are invalid. Inspection of `outputs/eval_v2/evaluation_clean_summary.json` shows:
- coupling_id was skipped: "train split has only one class: [1]"
- is_transition_window was skipped: "train split has only one class: [False]"
- load R²=1.0 indicates a data-leakage or degenerate label scenario

This dataset/split was not trajectory-level or had a construction error. Do not use or cite it.

---

## Key Lessons

- **Lesson 1**: TCN with temporal-neighbour contrastive + reconstruction does not extract
  factorised structure. Probes confirm this; high retrieval is misleading.

- **Lesson 2**: Hand-crafted FFT/summary baselines are strong. Any learned method must
  beat them by a meaningful margin *for the right reasons* before claiming success.

- **Lesson 3**: Retrieval metric alone is not a valid evaluation. Always report probes
  and clustering ARI alongside retrieval.

- **Lesson 4**: Load factor is never learned by any method tried so far. It requires an
  explicit objective. Spectral factor is learned partially (and could be extracted more simply
  by FFT).

- **Lesson 5**: Degenerate dataset splits produce meaningless high scores. Always run
  leakage checks and inspect whether all classes are present in each split.

---

## What NOT to Repeat

- TCN + temporal-neighbour contrastive with the same loss, expecting better factor recovery.
- Reporting only retrieval metrics.
- Using datasets without trajectory-level splitting and per-factor class balance checks.
- Citing the eval_v2 results with R²=1.0 or 94.7% accuracy.
