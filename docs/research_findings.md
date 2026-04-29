# Distilled Research Findings

Source inputs:
- context/output_1.txt.txt (analysis of current problems)
- experiments/rq1_recoverability/reports/result_tables/ (Phase 3 formal results)
- outputs/version2/reports/ (Phase 2 TCN/FSSS results)

For full context see [RETROSPECTIVE.md](../RETROSPECTIVE.md) and the
[per-phase history reports](../experiments/history/).

---

## Core Problem Summary

The current setup over-optimises local temporal similarity but underperforms on global mode
structure, which is required for mode-change detection and clustering. The training objective
is misaligned with the evaluation goal.

---

## Quantitative Results Summary

### Phase 3 — Best results on frs_clean_vnext_long (RBF probe balanced accuracy)

| Factor      | Baseline FFT | MOMENT (3 seeds) | TS2Vec e120 |
|-------------|-------------|------------------|-------------|
| mode_id     | 0.439       | 0.466 ±0.012     | **0.514**   |
| spectral_id | 0.696       | 0.756 ±0.018     | **0.767**   |
| coupling_id | 0.673       | 0.633 ±0.042     | **0.713**   |
| transition  | —           | 0.576 ±0.042     | 0.527       |
| mean_load R²| −0.006      | −0.037 ±0.049    | −0.006      |

**Avg factor (mode/spectral/coupling only)**: FFT 0.603, MOMENT 0.618, TS2Vec e120 **0.665**.
Gap over FFT baseline: +6.2pp for TS2Vec e120. Load: never learned by any method.

### Phase 3 — Clustering ARI (frs_clean_vnext_long)

| Factor      | MOMENT       | TS2Vec e80 | TS2Vec e120 |
|-------------|-------------|------------|-------------|
| mode_id     | 0.151        | 0.153      | **0.157**   |
| spectral_id | 0.180        | **0.207**  | 0.194       |
| coupling_id | 0.058        | 0.006      | 0.004       |

All ARI values are very low. More training (e80→e120) does not improve clustering.
Coupling ARI near zero despite coupling probe of 0.71 — probe and ARI dissociate.

### Phase 2 — Baseline vs TCN hybrid (FSSS test set, balanced accuracy)

| Factor      | Best Baseline | TCN Learned | Winner    |
|-------------|--------------|-------------|-----------|
| mode_id     | 0.277 (Summary) | 0.267    | Baseline  |
| spectral_id | 0.701 (Summary) | 0.653    | Baseline  |
| coupling_id | 0.413 (Summary) | 0.423    | Learned (marginal) |
| transition  | 0.558 (FFT)     | 0.534    | Baseline  |
| mean_load   | ≈0              | ≈0       | Tied      |

The TCN hybrid learned embedding did not beat hand-crafted baselines in Phase 2.

---

## Key Findings

### 1. Objective and evaluation mismatch
Training emphasises temporal-neighbour similarity, while evaluation expects factor-level
recovery and global mode structure. The model learns "what is nearby in time", not "what
shares the same underlying factors".

### 2. Positive-pair misalignment
Positive pairs are temporal neighbours, not same-factor samples across trajectories. The
model never learns to group windows from different trajectories that share the same mode.

### 3. Local structure dominates global structure
Contrastive + temporal loss enforces local smoothness but not global clustering. Retrieval
R@5 can be 0.96–0.99 (Phase 2) while clustering ARI is 0.03–0.13. These metrics are not
interchangeable.

### 4. Load factor is never learned
Load R² ≤ 0 for every method, every setting. No unsupervised method tried so far extracts
load information. An explicit regression objective is required.

### 5. Baselines are formidable
Hand-crafted FFT features achieve 60.3% average factor balanced accuracy (clean). The best
learned method (TS2Vec e120) achieves 66.5%. The gap (+6.2pp) is real but thin. Any new
method must beat FFT by a meaningful margin for the right reasons.

### 6. Probe and clustering ARI dissociate
TS2Vec e120 achieves coupling probe 0.713 but coupling clustering ARI 0.004. Probes measure
linear accessibility; ARI measures global geometric structure. Both must be reported.

### 7. More training improves probes, not clustering
TS2Vec e80 → e120 improves all probe scores but leaves ARI unchanged or worse. The training
objective is not producing globally separable structure—more training is not the solution.

### 8. MOMENT is more robust under noise
TS2Vec e120 clean→noisy degradation: −0.058 avg factor. MOMENT: −0.028. Pretrained
foundation models generalise better to noise than from-scratch trained encoders.

### 9. Transition windows are harder than stable regimes
Phase 2: mode accuracy drops from 28.2% (clean) to 22.3% (transition). Temporal smoothing
blurs transitions instead of modelling them as first-class events. This is a gap in both
the training objective and the evaluation protocol.

### 10. PLC logs and time series are a close fit
The target domain is PLC / SCADA logs: timestamped sensor readings (float) and machine state
codes (integer), no free text, no JSON. Parsing these to a multivariate time series matrix is
natural and largely lossless. The remaining gaps are engineering concerns: irregular per-channel
sampling rates, mixed discrete/continuous channel types, and named channel semantics. The FRS
benchmark is a valid scientific proxy. LLM text embedding is not a primary direction.

---

## Implications for Unified Roadmap

- Fix positive-pair design before adding more architecture complexity.
- Add explicit load regression auxiliary objective (first method to achieve R² > 0.1 is a
  genuine contribution).
- Require clustering ARI alongside probes in all evaluation reports.
- Require 3+ seeds for all method comparisons.
- Add OOD splits (held-out factor combinations) to the benchmark.
- Add segment-level evaluation (change-point detection, boundary F1).
- Define a synthetic PLC log generator and parse pipeline (Track B) to connect FRS to the
  real target domain (PLC / SCADA logs with sensor floats and integer state codes).

## Actionable Next Steps

See [docs/next_experiments.md](next_experiments.md) for the full prioritised roadmap.

Short list:
1. Factor-aware positive pairs (A1 — highest priority).
2. Multi-seed TS2Vec re-run (A4 — low effort, required for credibility).
3. Synthetic PLC log generator + parse pipeline (B1 + B2 — bridges FRS to real domain).
4. Load regression auxiliary objective (A2).
5. OOD splits (A3).
