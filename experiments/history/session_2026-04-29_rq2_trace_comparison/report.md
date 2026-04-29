# Session Report: 2026-04-29 — RQ2 Trace Comparison in Embedding Space

Status: **Implemented and submitted to cluster. Results pending.**
Code: [experiments/rq2_trace_comparison/](../../../experiments/rq2_trace_comparison/)
Docs: [docs/trace_comparison_idea.md](../../../docs/trace_comparison_idea.md)
Cluster job: `6536046` on Alvis (RUN_ID `RQ2-20260429-180416`)

---

## Session Goal

Design and implement a new experimental track (RQ2) that validates the core idea behind the
project at a geometrical level: does a good embedding produce a **coherent trace** in R^d when
the log/signal stream moves through known operating modes?

---

## Key Ideas Developed

### 1. PCA Worm Plot as a Sanity Check for Periodic Signals

For a periodic signal embedded via a sliding window, the first two PCA components capture the
dominant oscillation and its 90°-shifted quadrature. Plotting PC1 vs PC2 over time (a "worm
plot") traces a **closed loop** — a circle for a pure sine, a more complex closed curve for
non-sinusoidal or multi-component signals.

This is not a PCA assumption; it is a consequence of the structure of any periodic embedding.
Shape of the loop encodes harmonic content. Drift, non-closure, or fuzziness signals instability,
noise, or mode transition.

### 2. The Full Trace Object (not just 2D)

PCA is only a visualization tool. The actual object of interest is the full trace in R^d:

```
log stream → [window_1, ..., window_n] → [e_1, ..., e_n] ∈ R^d
```

Because the embedding is **deterministic**, the same log segment always maps to the same point.
This makes traces comparable across independent runs without fighting stochastic noise from the
embedding itself.

### 3. Known Mode Change Points Enable Trajectory Evaluation

If we know when the system switched operating mode (from ground truth: deployments, config
changes, incidents), we can:
- Segment the trace by mode label
- Compare same-mode traces across runs (do they occupy the same region?)
- Measure inter-mode separability and intra-mode consistency
- Detect transitions: the trace leaves one region and enters another

This gives a **trajectory-level quality criterion** for embeddings — stronger than per-window
probe accuracy because it tests global temporal coherence.

### 4. CNC Analogy

The modes are framed as CNC machine "parts": printing part A, B, C each has a characteristic
periodic axis-value pattern. Repeated prints of part A should produce approximately the same
loop in embedding space. This analogy grounds the abstract idea in a concrete, intuitive case.

---

## What Was Built

### `experiments/rq2_trace_comparison/` — Full Experiment Suite

#### Synthetic Data Generator
`src/rq2/generation/periodic_mode_generator.py`

Three datasets of increasing difficulty:

| Dataset | Channels | Noise σ | Mode separation |
|---------|----------|---------|-----------------|
| `p1_simple_1d` | 1 | 0.05 | High — clearly different frequencies |
| `p2_multichannel` | 4 | 0.08 | Medium — cross-channel frequency mixing |
| `p3_hard_noisy` | 4 | 0.20 | Low — similar frequencies, high noise |

Each dataset: 20 production runs, 7 mode segments per run (A→B→A→C→B→A→C), known change
points recorded. Window length 48, stride 12.

#### Trace Metrics
`src/rq2/evaluation/trace_metrics.py`

Five new metrics not present in the RQ1 evaluation suite:

| Metric | Description | Direction |
|--------|-------------|-----------|
| Mode Separability Index (MSI) | inter-mode centroid L2 / intra-mode std | ↑ |
| Loop Consistency (LC) | mean DTW across repeated same-mode traces | ↓ |
| Transition Sharpness (TS) | windows to cross mode midpoint at change points | ↓ |
| PCA Loop Compactness | convex hull area in PC1–PC2 (normalized) | ↓ |
| Centroid Stability | std of per-run mode centroids across test trajectories | ↓ |

DTW is implemented without external dependencies (pure numpy quadratic DP).

#### Visualization
`src/rq2/visualization/worm_plots.py`

Five plot types per experiment cell:
1. **PCA worm** — full production run, colored by time, mode-colored scatter
2. **Mode loop overlay** — all repetitions of each mode overlaid, 1σ ellipses
3. **Centroids panel** — per-mode centroid + confidence ellipse
4. **Centroid distance over time** — scalar distance to each mode centroid, change points marked
5. **Pairwise distance heatmap** — inter-mode centroid L2 distance matrix

#### Pipeline Scripts
Sequential 6-step pipeline: `scripts/01_generate_periodic_modes.py` through `06_build_report.py`

Step 02 reuses `src/version2/evaluation/baseline_features.py` (FFT features, no new code).
Step 03 reuses the `TS2VecStyleEncoder` architecture from `experiments/rq1_recoverability/scripts/11_run_ts2vec_recoverability.py`.

#### Embeddings Evaluated (9 experiment cells = 3 problems × 3 models)

| Model | Type | Training |
|-------|------|----------|
| FFT baseline | Deterministic | None |
| MOMENT (AutonLab/MOMENT-1-base) | Pretrained foundation model | Frozen |
| TS2Vec (TCN dilated residual) | Self-supervised | Per dataset |

#### SLURM Job
`slurm/rq2_full.slurm` — 1h30m, T4:1 GPU, sequential pipeline in one job.

---

## Documentation Created

- `docs/trace_comparison_idea.md` — conceptual write-up of the trace comparison idea, metrics,
  connection to the existing RQ1 benchmark, and next steps.

---

## Cluster Submission

First attempt (job `6536003`) failed after 20s: `momentfm` not installed in the Apptainer
container (`codeserver-PyTorch-2.9.1.sif`). Steps 1 and 2a (data generation + FFT embeddings)
completed successfully. Data is on Alvis at:
```
/cephyr/users/silvan/Alvis/log-to-vec-benchmark/experiments/rq2_trace_comparison/data/
/cephyr/users/silvan/Alvis/log-to-vec-benchmark/experiments/rq2_trace_comparison/embeddings/*_fft.npz
```

Fix: added `pip install --user momentfm` at job startup and skip-if-exists guards for steps 1
and 2a.

Second attempt (job `6536046`, RUN_ID `RQ2-20260429-180416`) submitted. Status at session end:
PENDING (Priority). Expected wall time ~1h once it starts.

---

## Expected Outputs (when job completes)

```
experiments/rq2_trace_comparison/
  metrics/   {p1,p2,p3}_{fft,moment,ts2vec}.json    — 9 metric files
  plots/     {problem}/{model}/*.png                 — 5 plots × 9 cells = 45 plots
  reports/   report.md                               — summary table + plot refs
             summary.json                            — machine-readable flat table
```

---

## Sanity Expectations

- P1/FFT: very high MSI — frequencies are trivially separable by spectrum
- P1 worm plots: visually distinct closed loops per mode
- P3: lowest MSI across all models — hard problem by design
- TS2Vec on P3: should improve over FFT on MSI (learned robustness)
- MOMENT on P1: may underperform FFT (pretrained on different domain, 1-channel input)

---

## Next Steps

1. Collect results once job `6536046` completes:
   ```bash
   rsync -az alvis1:.../experiments/rq2_trace_comparison/{metrics,plots,reports}/ \
         experiments/rq2_trace_comparison/local_results/
   ```
2. Read `reports/report.md` and fill in the **Key Findings** section
3. If MSI is too low across all models on P3, consider adding a fourth model:
   supervised contrastive (uses mode labels during training) as an upper bound
4. If worm plots look clean: extend to real log data using the log→FRS bridge (Track C)
5. Consider adding Fréchet distance as an alternative to DTW for LC metric
