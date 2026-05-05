# RQ2 — Trace Comparison in Embedding Space: Report

## Overview

Three synthetic periodic-mode datasets (CNC-analogy: parts A, B, C).
Each mode is a characteristic multi-channel sine wave pattern.
Embeddings compared: FFT baseline (deterministic), MOMENT (pretrained), TS2Vec (trained).
Key question: **does the embedding produce geometrically coherent, mode-separable traces?**

Metrics:
- **MSI** (Mode Separability Index) — inter-mode distance / intra-mode spread. ↑ better.
- **Loop Consistency** — DTW distance between repeated traces of the same mode. ↓ better.
- **Transition Sharpness** — windows to cross midpoint after a mode change. ↓ better.
- **PCA Compactness** — convex hull area of mode loop in PC1–PC2. ↓ better.
- **Centroid Stability** — std of per-run mode centroids across test trajectories. ↓ better.

## Problem 1 — Simple 1D (1 channel, clearly separated frequencies)

| Model | MSI ↑ | Loop Consistency (DTW) ↓ | Transition Sharpness (windows) ↓ | PCA Compactness ↓ | Centroid Stability ↓ |
| --- | --- | --- | --- | --- | --- |
| **fft** | 11.097 | 66.579 | **3.000** | 0.499 | 0.289 |
| **moment** | **71.109** | **2.757** | 5.667 | **0.231** | **0.002** |
| **ts2vec** | 20.001 | 44.573 | 3.000 | 0.373 | 0.085 |

**Selected plots (worm plot, one test trajectory):**

- `fft` worm: `experiments/rq2_trace_comparison/plots/p1_simple_1d/fft/p1_simple_1d_fft_worm.png`
- `fft` loops: `experiments/rq2_trace_comparison/plots/p1_simple_1d/fft/p1_simple_1d_fft_mode_loops.png`
- `moment` worm: `experiments/rq2_trace_comparison/plots/p1_simple_1d/moment/p1_simple_1d_moment_worm.png`
- `moment` loops: `experiments/rq2_trace_comparison/plots/p1_simple_1d/moment/p1_simple_1d_moment_mode_loops.png`
- `ts2vec` worm: `experiments/rq2_trace_comparison/plots/p1_simple_1d/ts2vec/p1_simple_1d_ts2vec_worm.png`
- `ts2vec` loops: `experiments/rq2_trace_comparison/plots/p1_simple_1d/ts2vec/p1_simple_1d_ts2vec_mode_loops.png`

## Problem 2 — Multi-channel (4ch, cross-channel frequency mixing)

| Model | MSI ↑ | Loop Consistency (DTW) ↓ | Transition Sharpness (windows) ↓ | PCA Compactness ↓ | Centroid Stability ↓ |
| --- | --- | --- | --- | --- | --- |
| **fft** | 22.457 | 146.132 | 3.333 | **0.262** | 0.278 |
| **moment** | **27.186** | **1.425** | **1.667** | 1.627 | **0.001** |
| **ts2vec** | 4.426 | 108.180 | 3.667 | 48.396 | 0.509 |

**Selected plots (worm plot, one test trajectory):**

- `fft` worm: `experiments/rq2_trace_comparison/plots/p2_multichannel/fft/p2_multichannel_fft_worm.png`
- `fft` loops: `experiments/rq2_trace_comparison/plots/p2_multichannel/fft/p2_multichannel_fft_mode_loops.png`
- `moment` worm: `experiments/rq2_trace_comparison/plots/p2_multichannel/moment/p2_multichannel_moment_worm.png`
- `moment` loops: `experiments/rq2_trace_comparison/plots/p2_multichannel/moment/p2_multichannel_moment_mode_loops.png`
- `ts2vec` worm: `experiments/rq2_trace_comparison/plots/p2_multichannel/ts2vec/p2_multichannel_ts2vec_worm.png`
- `ts2vec` loops: `experiments/rq2_trace_comparison/plots/p2_multichannel/ts2vec/p2_multichannel_ts2vec_mode_loops.png`

## Problem 3 — Hard / Noisy (4ch, similar frequencies, σ=0.20)

| Model | MSI ↑ | Loop Consistency (DTW) ↓ | Transition Sharpness (windows) ↓ | PCA Compactness ↓ | Centroid Stability ↓ |
| --- | --- | --- | --- | --- | --- |
| **fft** | **20.200** | 167.400 | 4.000 | **0.332** | 0.364 |
| **moment** | 14.336 | **1.645** | **1.333** | 7.967 | **0.001** |
| **ts2vec** | 4.562 | 107.795 | 4.000 | 55.869 | 0.529 |

**Selected plots (worm plot, one test trajectory):**

- `fft` worm: `experiments/rq2_trace_comparison/plots/p3_hard_noisy/fft/p3_hard_noisy_fft_worm.png`
- `fft` loops: `experiments/rq2_trace_comparison/plots/p3_hard_noisy/fft/p3_hard_noisy_fft_mode_loops.png`
- `moment` worm: `experiments/rq2_trace_comparison/plots/p3_hard_noisy/moment/p3_hard_noisy_moment_worm.png`
- `moment` loops: `experiments/rq2_trace_comparison/plots/p3_hard_noisy/moment/p3_hard_noisy_moment_mode_loops.png`
- `ts2vec` worm: `experiments/rq2_trace_comparison/plots/p3_hard_noisy/ts2vec/p3_hard_noisy_ts2vec_worm.png`
- `ts2vec` loops: `experiments/rq2_trace_comparison/plots/p3_hard_noisy/ts2vec/p3_hard_noisy_ts2vec_mode_loops.png`

## Key Findings

*(Fill in after running experiments.)*

Expected sanity checks:
- P1 FFT should show very high MSI (frequencies trivially separable by spectrum)
- P3 should have lowest MSI across all models (similar freqs + high noise)
- PCA worm for P1/FFT should show visually distinct closed loops per mode
- TS2Vec trained on P3 should improve over FFT on MSI, demonstrating learned robustness
