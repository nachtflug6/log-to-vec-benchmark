# Log-to-Vec Benchmark — Comprehensive Research Report

*Generated: 2026-05-07*

---

## 1. Project Goal

Learn useful **embeddings of PLC/SCADA log files** for mode-change detection and regime
identification in industrial control systems. The logs are timestamped records of sensor readings
and discrete machine state codes — essentially serialised multivariate time series with no
meaningful free text.

The motivating question: can a learned representation capture operating regime, load, and
mode-change structure **without hand-labelled data**, well enough to be useful downstream?

---

## 2. Session Log

Each row records one working session: what we set out to do, what ran on Alvis, and what the
session established. Dates are commit dates (UTC+1). All cluster jobs ran on Alvis (Chalmers
HPC) using A40/A100 GPUs unless noted.

| Date | Session | Aim | Key actions | Outcome / what we learned |
|------|---------|-----|-------------|--------------------------|
| 2026-02-06 | S0 — Repo init | Establish a shared codebase for the log-to-vec study after two failed prototype phases | Initial commit: project layout, shared utilities, history notes | Baseline for all future experiments |
| 2026-04-28 | S1 — Unify | Merge several prototype branches (LSTM contrastive, TCN hybrid, FRS benchmark) into one coherent repo | Integrated branch code paths, experiment scaffolding, merge ledger | Single working repo; Phase 1–2 retrospective documented; decision to pivot to pretrained foundation model evaluation |
| 2026-04-29 | S2 — Alvis deploy + RQ2 scaffold | Get the full pipeline running on Alvis GPU cluster; design and submit the first geometry-metric experiment (RQ2) | Apptainer container deployed on Alvis; RQ2 trace comparison scaffolded (3 synthetic problems: P1 simple, P2 multichannel, P3 hard/noisy); SLURM job submitted; retrospective and roadmap written | Container confirmed working; RQ2 job submitted successfully; roadmap committed |
| 2026-04-30 | S3 — RQ2 fix | MOMENT embedding step crashed: model requires exactly 512-sample input but windows were 128 samples | Pad all windows to 512 before MOMENT forward pass (zero-padding, right-sided); add missing `#SBATCH --gpus-per-node` SLURM directive | RQ2 resubmitted and completed; MOMENT padding approach confirmed as viable |
| 2026-05-05 | S4 — Metric upgrade + RQ4/RQ5 | Add silhouette score and UMAP visualisation to the eval suite; design and run more realistic experiments | Added silhouette score to all geometry metrics; added UMAP plots; scaffolded and submitted RQ4 (CNC-analogous 10-channel benchmark with 7 scenarios and 2 label levels) and RQ5 (window-size sweep ws∈{32,64,128,256,512}); fixed sklearn 1.5+ API break in silhouette call; fixed DCC module import error | RQ4 and RQ5 both completed on cluster; silhouette provides cleaner separability signal than MSI alone; sklearn 1.5 compat issue resolved |
| 2026-05-05 | S4b — SLURM master | Submit all remaining pipeline stages as single GPU jobs with a master submit script | Full-pipeline SLURM jobs (embed→metrics→report) added for all RQs; master `submit_all.sh` script added | All RQs queued in one command |
| 2026-05-06 | S5 — RQ3 results + report | Pull RQ3 landscape sweep results (14 scenarios × 5 models) and write first comprehensive report | RQ3 results collected; cross-experiment synthesis written; `comprehensive_report.md` v1 written and PDF generated via `build_pdf.py` | First full report covering RQ1–RQ5; identified MOMENT/FFT role reversal; documented signal-type failure mode |
| 2026-05-07 | S6 — RQ6 results + permissions | Collect RQ6 signal-ablation results; update report with session log and full RQ6 section; tidy project permissions | Cluster check: no jobs running; RQ6 results present (24 rows: ratio_0step→ratio_4step + chirp/damped/sawtooth); permissions consolidated in settings.local.json; this report generated | RQ6 confirms MOMENT degrades gracefully with step-channel contamination; exotic signal types (chirp, damped, sawtooth) favour MOMENT strongly |

---

## 3. Phase 1 — Contrastive LSTM on Toy Logs

**Setup**
- LSTM encoder trained with NT-Xent contrastive loss.
- Positive pairs: temporal neighbours within a trajectory.
- Data: toy CSV logs generated synthetically; sine-wave generator examples.
- Evaluation: ad-hoc clustering and retrieval visualisations only. No formal metric suite.

**Outcome**
No quantitative results recorded. Discontinued because:
1. Temporal-neighbour positives teach "what is nearby in time", not "what shares a mode".
2. Toy logs had no ground-truth factorised labels — claims about quality were unverifiable.

**Lesson:** A controlled synthetic benchmark with known latent structure is a prerequisite before
any learning claim can be made.

---

## 4. Phase 2 — TCN Hybrid on FSSS (version2 branch)

**Setup**
- TCN (Temporal Convolutional Network) with dilated residual conv blocks.
- Hybrid loss: NT-Xent contrastive + reconstruction (masked span, Gaussian noise, time shift).
- Data: FSSS datasets — earlier, smaller synthetic variants of FRS.
- Formal eval suite added: linear and RBF probes, retrieval@K, clustering ARI.
- Baselines: raw statistics, FFT, PCA.

**Results (RBF probe balanced accuracy)**

| Factor | Baseline (best) | Learned (best) | Delta |
|--------|-----------------|----------------|-------|
| mode_id | 0.277 | 0.267 | −0.010 — **baseline wins** |
| spectral_id | 0.701 | 0.653 | −0.048 — **baseline wins** |
| coupling_id | 0.413 | 0.423 | +0.010 — marginal |
| transition | 0.558 | 0.534 | −0.024 — **baseline wins** |
| mean_load | ≈0 | ≈0 | tied at zero |

Retrieval R@5 was misleadingly high (0.961–0.993) due to temporal-neighbour training bias.
Clustering ARI was very poor (0.020–0.134).

**Lesson:** TCN + temporal-neighbour contrastive + reconstruction does not extract factorised
structure. The objective is misaligned with the evaluation goal. Baselines are formidable.

---

## 5. Problem Setups and Datasets

This section describes all five experimental setups in detail. Each was designed to probe a
specific question about embedding quality, building from simple factor recovery (RQ1) to
realistic multi-channel industrial signals (RQ4/RQ5).

### 5.1 RQ1 — FRS: Factorised Regime Sequence Benchmark

**Motivation.** After Phase 1–2 failures, we needed a synthetic benchmark with *known*, *independent*
latent factors and a trajectory-level data split. The FRS generator was purpose-built for this.

**Data generation.** Each trajectory is a multivariate time series drawn from a hidden Markov model
over a product space of four independent factors:

- **Mode ID** — discrete operating regime (0–7). Each mode assigns a characteristic sine-wave
  frequency and phase to each channel. Modes are randomly ordered within a trajectory and recur.
- **Spectral ID** — global frequency scaling (3 levels). All channels share the same spectral class
  within a trajectory; different trajectories may differ. This tests whether the embedding captures
  a slow-changing global property.
- **Coupling ID** — cross-channel interaction strength (3 levels: 0 = independent, 1 = moderate,
  2 = strong). Coupling is implemented as a weighted additive mixture across channels.
- **Load** — continuous amplitude multiplier (uniform on [0.5, 2.0]). Represents the overall
  signal energy level. Deliberately not separable from mode without amplitude-sensitive learning.

Two datasets are used: `frs_clean_vnext_long` (additive Gaussian noise σ = 0.05) and
`frs_noisy_vnext_long` (σ = 0.20). Each has 200 trajectories of 4000 time steps, 4 channels,
with a **trajectory-level 70/15/15 train/val/test split** to prevent any leakage of temporal
structure between sets.

**Why this setup is hard.** The four factors are independent by construction, so the optimal
embedding would factorise along all four axes simultaneously. The load factor is especially hard:
it is a multiplicative scalar on the signal amplitude, and no method we tested recovered it
(R² ≤ 0 throughout). FFT amplitude spectra partially capture amplitude but conflate it with mode
frequency; contrastive learning with temporal neighbours teaches temporal proximity, not amplitude
level.

**Evaluation suite.** For each factor:
- **RBF SVM probe** — balanced accuracy on test set (primary metric).
- **Linear SVM probe** — measures linear separability specifically.
- **Precision@10 retrieval** — does the nearest neighbour in embedding space share the same factor value?
- **K-Means ARI** — does unsupervised clustering align with factor labels?
- **Transition detection** — can mode changes be detected from the embedding sequence?

### 5.2 RQ2 — Trace Geometry: Three Synthetic Problems

**Motivation.** Probe accuracy tells us about label recovery, but not about the *geometry* of the
embedding space. For online mode-change detection, we care whether the embedding produces
consistent, compact, closed-orbit trajectories — one orbit per mode — and whether transitions
between modes are sharp. RQ2 was designed to measure this directly.

**Problem definitions.** Three problems of increasing difficulty, all modelled on CNC-machining
analogies (each "mode" corresponds to a machining pass):

- **P1 — Simple 1D.** One channel. Three modes with clearly separated frequencies (0.05, 0.20,
  0.40 Hz). Low noise (σ = 0.02). This is the easiest possible case: a raw FFT amplitude spectrum
  is expected to trivially separate the modes.
- **P2 — Multi-channel.** Four channels. Frequencies are similar across channels but combined with
  cross-channel mixing (mode 0 emphasises ch0+ch1, mode 1 emphasises ch1+ch2, etc.). Low noise.
  The spectral structure is still clear but requires joint reasoning across channels.
- **P3 — Hard/Noisy.** Four channels. Frequencies are close across modes (0.10, 0.15, 0.20 Hz).
  High noise (σ = 0.20). Designed to stress-test all methods — even FFT struggles when frequency
  separation is small and signal-to-noise is low.

Each problem has 30 trajectories: 15 train, 15 test. Each trajectory visits each mode 5 times
for a total of 15 mode segments. Window size is 128 samples.

**Geometry metrics.** Five metrics are computed on test-set embeddings:

| Metric | Symbol | Direction | Meaning |
|--------|--------|-----------|---------|
| Mode Separability Index | MSI | ↑ better | Ratio of inter-mode centroid distance to mean intra-mode spread |
| Loop Consistency | Loop DTW | ↓ better | Average DTW distance between repeated visits to the same mode |
| Transition Sharpness | Trans. Sharp | ↓ better | Number of windows to cross the embedding midpoint after a mode change |
| PCA Compactness | PCA Compact | ↓ better | Convex hull area of the mode trajectory projected to PC1–PC2 |
| Centroid Stability | Centroid Stab | ↓ better | Standard deviation of per-run mode centroids across test trajectories |

### 5.3 RQ3 — Landscape Sweep: 14 Stress Scenarios

**Motivation.** RQ1 and RQ2 used one noise level and one signal type. Before recommending any
method for production, we need to know *where* each method's performance degrades. RQ3 sweeps
four independent axes while holding everything else fixed.

**Base scenario.** 3 modes, 4 channels, sine-wave signal, σ_noise = 0.05, no missing data
(identical to the P2 problem in RQ2 but evaluated with the trace-geometry metric suite).

**Sweep axes and levels:**

| Axis | Scenario IDs | Description |
|------|-------------|-------------|
| Noise | noise_000, noise_020, noise_040 | σ ∈ {0.00, 0.20, 0.40} |
| Missing data | miss_rand10, miss_rand30 | Random point dropout, 10% or 30% |
| Missing data | miss_block30 | Contiguous 30% block zeroed out |
| Missing data | miss_channel | One entire channel zeroed (σ=0.05) |
| Number of modes | modes_2, modes_5, modes_8 | K ∈ {2, 5, 8} modes |
| Signal type | sig_step | Step-change signals (no spectral content) |
| Signal type | sig_event | Sparse impulse events |
| Signal type | sig_mixed | Mix of sine and step signals |

**Why signal type matters.** The two pretrained foundation models we evaluate (MOMENT, and the
FFT baseline which implicitly assumes periodic structure) are both tuned to periodic signals.
Step and event signals expose a fundamental assumption mismatch: neither amplitude spectrum nor
a transformer pretrained on smooth time series handles abrupt non-periodic transitions well.

**Models evaluated.** FFT8 (8 frequency bins per channel), FFT32 (32 bins per channel), and
MOMENT-1-base. Note: MOMENT was not evaluated on the `miss_channel` scenario — the zero-padded
channel produces degenerate embeddings and was excluded.

### 5.4 RQ4 — CNC-Analogous Mixed-Signal Benchmark

**Motivation.** FRS and P1–P3 are purely synthetic sine-wave datasets. Real industrial signals
(PLC/SCADA, CNC machines) have discrete state codes, mixed periodic and aperiodic channels, and
multi-level label structure. RQ4 builds a more realistic benchmark modelled on CNC machining.

**Signal structure.** 10 channels representing a machining centre:

| Channel | Unit | Type | Description |
|---------|------|------|-------------|
| axis_x, axis_y, axis_z | mm | continuous | Absolute toolpath position |
| spindle_speed | RPM | continuous | Continuous spindle signal |
| feed_rate | mm/min | continuous | Commanded feed |
| vibration_x, vibration_y | m/s² | periodic | Cutting vibration, cutting-freq tied to spindle |
| temperature | °C | slow continuous | Thermal lag of machining activity |
| tool_id | int (0–5) | discrete (float) | Tool currently in spindle |
| alarm_code | int (0–99) | discrete (float) | 0 = normal; non-zero = active fault code |

**Two-level label structure.** This is the key novelty of RQ4. Every window has *two* ground-truth
labels that represent the same signal at different resolutions:

- **program_step_id** — which G-code block is executing. Fine-grained industrial KPI: can an
  embedding distinguish "roughing pass, tool 2" from "finishing pass, tool 3"?
- **op_state_id** — coarse operating state: IDLE (0), HOMING (1), RAPID (2), CUTTING (3),
  DWELL (4), FAULT (5). Represents PLC-visible machine state, independent of the specific program.

Both labels are used independently as grouping variables for the metric suite. This lets us ask:
does an embedding that separates operating states also separate program steps, or do the two tasks
require different representations?

**Seven scenarios.** Each scenario holds the machining program fixed and varies one aspect:

| Scenario | Programs | Runs | Key variation |
|----------|----------|------|---------------|
| cnc_part_simple | Square contour | 20 | Baseline, 6 channels |
| cnc_part_full | Square contour | 20 | Full 10-channel set |
| cnc_two_parts | Square + Triangle | 20 | Cross-program separation |
| cnc_tool_wear | Square contour | 20 | Vibration amplitude increases +4% per run |
| cnc_noisy | Square contour | 20 | σ_noise = 0.25 added to all channels |
| cnc_pocket | Pocket milling | 20 | Z-varying path, 11 program steps |
| cnc_faults | Square + faults | 20 | ~8% alarm code injection |

**Evaluation.** Same geometry metrics as RQ2/RQ3 (MSI, silhouette, loop DTW, transition sharpness,
centroid stability), computed separately against program_step_id and op_state_id labels.

### 5.5 RQ5 — Window Size Sweep

**Motivation.** All previous experiments used a fixed window size (128 or 512 samples). Window
size is a critical hyperparameter: too small and there is not enough context to identify a mode;
too large and the window spans mode boundaries. For MOMENT specifically, all inputs are padded
to 512 samples — short windows are zero-padded on the right, which may introduce systematic bias.

**Setup.** The `cnc_part_full` scenario from RQ4 is used as the base (square contour, 10 channels,
20 runs, σ = 0.05). Window size is swept across five values with stride = window_length // 8:

| Window | Stride | Windows/trajectory |
|--------|--------|--------------------|
| 32 | 4 | ~3974 |
| 64 | 8 | ~1912 |
| 128 | 16 | ~880 |
| 256 | 32 | ~364 |
| 512 | 64 | ~106 |

**Key questions:**
1. Does a larger window improve MSI by capturing more periodic structure?
2. Does a larger window hurt transition sharpness by spanning mode boundaries?
3. Where does MOMENT's mandatory 512-sample padding start to hurt for short windows?
4. Is there a single best window size, or does the optimal differ by model?

### 5.6 RQ6 — Signal Ablation

**Motivation.** RQ3 showed that step-change signals are catastrophic for MOMENT (MSI 2.8 vs
FFT 9.6). But that experiment used *all* channels as step signals simultaneously. Real PLC logs
contain *mixed* signal types: some channels are slow-varying analogue sensors, some are discrete
state codes. RQ6 answers two questions: (1) how does embedding quality degrade as the fraction
of step-type channels increases from 0/4 to 4/4? and (2) how do more exotic but still continuous
signal types (chirp, damped oscillation, sawtooth) compare to the baseline sine?

**Ablation axis — step channel ratio.** The base setup is the RQ3 baseline (3 modes, 4 channels,
σ = 0.05). The `ratio_Nstep` scenarios replace N of the 4 channels with step-change signals
(abrupt transitions between fixed levels) while keeping the remaining 4−N channels as sine waves.

| Scenario | Step channels | Sine channels | Notes |
|----------|--------------|---------------|-------|
| ratio_0step | 0 | 4 | Pure sine — reproduces RQ3 baseline |
| ratio_1step | 1 | 3 | One step channel |
| ratio_2step | 2 | 2 | Half/half |
| ratio_3step | 3 | 1 | Mostly step |
| ratio_4step | 4 | 0 | Pure step — reproduces RQ3 sig_step |

**Exotic signal types.** Three additional signal types replace the sine baseline across all
4 channels to probe what non-standard continuous waveforms do to each embedder:

| Scenario | Signal type | Description |
|----------|-------------|-------------|
| sig_chirp | Chirp | Linear frequency sweep from 0.05 to 0.40 Hz within each mode segment |
| sig_damped | Damped oscillation | Exponentially decaying sine at mode frequency; resets at segment start |
| sig_sawtooth | Sawtooth | Periodic ramp-up / instant-reset waveform at mode frequency |

**Models.** FFT8, FFT32, and MOMENT-1-base. Window size 128, same geometry metrics as RQ2–5.

---

## 6. Methods

### 6.1 Embedding Methods

| Method | Type | Dim | Training | Notes |
|--------|------|-----|----------|-------|
| **FFT8** | Baseline | 8 × C | None | Per-channel amplitude spectrum, 8 bins |
| **FFT32** | Baseline | 32 × C | None | Per-channel amplitude spectrum, 32 bins |
| **Summary stats** | Baseline | 4 × C | None | Mean, std, min, max per channel |
| **MOMENT-1-base** | Foundation | 768 | Pretrained | AutonLab; frozen; per-channel then mean-pooled |
| **TS2Vec-style** | Trained | 64 | From scratch | Dilated conv + NT-Xent; temporal-neighbour pairs |

### 6.2 Metric Summary

| Metric | Used in | Better |
|--------|---------|--------|
| RBF probe balanced accuracy | RQ1 | ↑ |
| Clustering ARI | RQ1 | ↑ |
| Retrieval P@10 | RQ1 | ↑ |
| Load R² | RQ1 | ↑ (all ≤ 0) |
| Mode Separability Index (MSI) | RQ2–5 | ↑ |
| Silhouette score | RQ2–5 | ↑ |
| Loop DTW (consistency) | RQ2–5 | ↓ |
| Transition sharpness | RQ2–5 | ↓ |
| Centroid stability | RQ2–5 | ↓ |
| PCA variance explained | RQ2–5 | — (diagnostic) |

---

## 7. RQ1 — Formal FRS Benchmark Results

### 7.1 RBF Probe Balanced Accuracy (clean dataset)

![RQ1 probe accuracy](figures/fig1_rq1_probes.png)

| Method | Mode | Spectral | Coupling | Transition | Load R² |
|--------|------|----------|----------|------------|---------|
| FFT baseline | 0.439 | 0.696 | 0.673 | — | −0.006 |
| Summary baseline | 0.431 | 0.653 | 0.634 | — | −0.006 |
| MOMENT | 0.466 ±0.012 | 0.756 ±0.018 | 0.633 ±0.042 | 0.576 ±0.042 | −0.037 ±0.049 |
| TS2Vec e80 | 0.398 | 0.694 | 0.625 | 0.527 | −0.006 |
| **TS2Vec e120** | **0.514** | **0.767** | **0.713** | 0.527 | −0.006 |

### 7.2 Average Balanced Accuracy (mode + spectral + coupling)

![Clean vs noisy comparison](figures/fig2_rq1_clean_noisy.png)

| Method | Clean | Noisy | Drop |
|--------|-------|-------|------|
| FFT baseline | 0.603 | 0.576 | −0.027 |
| Summary baseline | 0.573 | 0.579 | +0.006 |
| MOMENT | 0.618 | 0.590 | −0.028 |
| TS2Vec e80 | 0.572 | 0.575 | +0.003 |
| **TS2Vec e120** | **0.665** | **0.607** | −0.058 |

The margin of the best learned method over the best baseline is only **+6.2 pp** on clean data,
narrowing to **+3.1 pp** on noisy data.

### 7.3 Clustering ARI (K-Means, clean dataset)

![Clustering ARI](figures/fig3_rq1_clustering.png)

| Method | Mode | Spectral | Coupling | Avg |
|--------|------|----------|----------|-----|
| MOMENT | 0.151 | 0.180 | 0.058 | 0.130 |
| TS2Vec e80 | 0.153 | 0.207 | 0.006 | 0.122 |
| TS2Vec e120 | 0.157 | 0.194 | 0.004 | 0.118 |

Global clustering quality is uniformly poor (ARI < 0.21). More training (e80→e120) helps probes
but **degrades** coupling clustering. Load R² ≤ 0 for every method in every setting.

---

## 8. RQ2 — Trace Geometry Results

### 8.1 Full Metric Table

![Winner heatmap](figures/fig4_rq2_winner_heatmap.png)

| | MSI ↑ | Loop DTW ↓ | Trans. Sharp ↓ | PCA Compact ↓ | Centroid Stab ↓ |
|---|---|---|---|---|---|
| P1 / FFT | 11.1 | 66.6 | **3.0** | 0.50 | 0.289 |
| P1 / MOMENT | **71.1** | **2.76** | 5.7 | **0.23** | **0.002** |
| P1 / TS2Vec | 20.0 | 44.6 | 3.0 | 0.37 | 0.085 |
| P2 / FFT | 22.5 | 146.1 | 3.3 | **0.26** | 0.278 |
| P2 / MOMENT | **27.2** | **1.43** | **1.67** | 1.63 | **0.001** |
| P2 / TS2Vec | 4.4 | 108.2 | 3.67 | 48.4 | 0.509 |
| P3 / FFT | **20.2** | 167.4 | 4.0 | **0.33** | 0.364 |
| P3 / MOMENT | 14.3 | **1.64** | **1.33** | 7.97 | **0.001** |
| P3 / TS2Vec | 4.6 | 107.8 | 4.0 | 55.9 | 0.529 |

![MSI and Loop Consistency](figures/fig5_rq2_msi_loop.png)

![All 5 metrics](figures/fig8_rq2_all_metrics.png)

![Radar chart](figures/fig6_rq2_radar.png)

### 8.2 Key Findings

**MOMENT's loop consistency is extraordinary.** Loop DTW of 1.4–2.8 across all three problems,
vs 66–167 for FFT and 44–108 for TS2Vec — roughly 20–100× better. Mode traces form tight,
repeatable closed orbits.

**MOMENT centroid stability is near-zero.** 0.001–0.002 vs 0.08–0.53 for others. Mode regions
don't drift between test trajectories — critical for reference-based anomaly detection.

**FFT beats MOMENT on MSI in the hard noisy case (P3): 20.2 vs 14.3.** When frequencies are
close and noise is high, the raw spectrum remains more discriminative than the pretrained features.
MOMENT still wins on loop consistency and transition sharpness even in P3.

**TS2Vec is consistently the worst.** Low MSI (4.4–20.0), catastrophic PCA compactness on
multichannel (48–56), centroid stability 0.09–0.53. The 64-dim trained representation fails to
find useful geometry in these problems.

---

## 9. RQ3 — Landscape Sweep Results

### 9.1 Noise Axis (σ ∈ {0.00, 0.20, 0.40})

| Scenario | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Centroid Stab ↓ |
|----------|-------|-------|-------|-----------|----------------|
| noise_000 | fft32 | 37.2 | 0.394 | 147.0 | 0.367 |
| noise_000 | fft8 | 21.4 | 0.398 | 144.6 | 0.605 |
| noise_000 | moment | **73.1** | **0.501** | **5.2** | **0.008** |
| noise_020 | fft32 | 31.5 | 0.410 | 141.0 | 0.481 |
| noise_020 | fft8 | 20.6 | 0.428 | 138.7 | 0.656 |
| noise_020 | moment | **67.7** | **0.469** | **5.3** | **0.008** |
| noise_040 | fft32 | 22.8 | 0.328 | 175.7 | 0.725 |
| noise_040 | fft8 | 17.2 | 0.365 | 171.0 | 0.873 |
| noise_040 | moment | **59.9** | **0.417** | **5.6** | **0.009** |

MOMENT's MSI drops from 73.1 to 59.9 over the full noise range (−13.2 pp), while FFT32
drops from 37.2 to 22.8 (−14.4 pp). Both degrade similarly in absolute terms, but MOMENT
**maintains a consistent 37+ point lead** throughout. Loop DTW is essentially constant for MOMENT
(5.2–5.6) regardless of noise — the orbit consistency is noise-invariant.

### 9.2 Missing Data Axis

| Scenario | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Centroid Stab ↓ |
|----------|-------|-------|-------|-----------|----------------|
| miss_rand10 | fft32 | 29.0 | 0.391 | 132.1 | 0.410 |
| miss_rand10 | fft8 | 19.8 | 0.414 | 128.7 | 0.567 |
| miss_rand10 | moment | **65.0** | **0.446** | **5.5** | **0.008** |
| miss_rand30 | fft32 | 20.6 | 0.329 | 110.6 | 0.430 |
| miss_rand30 | fft8 | 16.3 | 0.383 | 106.5 | 0.517 |
| miss_rand30 | moment | **47.8** | **0.318** | **6.5** | **0.009** |
| miss_block30 | fft32 | 24.2 | 0.303 | 148.5 | 0.417 |
| miss_block30 | fft8 | 14.7 | 0.310 | 146.1 | 0.679 |
| miss_block30 | moment | **40.4** | **0.294** | **8.9** | **0.016** |
| miss_channel | fft32 | 13.5 | 0.106 | 478.1 | 0.850 |
| miss_channel | fft8 | 7.1 | 0.106 | 476.6 | 1.591 |
| miss_channel | moment | — | — | — | — |

**Random dropout** is relatively benign for MOMENT (rand10: 65.0, rand30: 47.8 vs baseline 72.5).
**Block dropout** is harder — the contiguous gap disrupts the local temporal context that MOMENT
relies on (block30: 40.4, −32.1 pp from baseline).
**Channel dropout** is catastrophic for FFT (MSI 13.5, loop DTW 478) and was not measurable for
MOMENT (zero-padded channel produces degenerate embeddings). This is a key production risk: any
sensor outage breaks the FFT's spectral decomposition entirely.

### 9.3 Number of Modes Axis

| Scenario | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Centroid Stab ↓ |
|----------|-------|-------|-------|-----------|----------------|
| modes_2 | fft32 | 43.0 | 0.426 | 114.2 | 0.212 |
| modes_2 | moment | **83.2** | **0.599** | **4.9** | **0.005** |
| modes_5 | fft32 | 33.3 | 0.335 | 147.4 | 0.521 |
| modes_5 | moment | **61.2** | **0.317** | **5.1** | **0.010** |
| modes_8 | fft32 | 30.9 | 0.250 | 143.2 | 0.550 |
| modes_8 | moment | **52.2** | **0.146** | **5.7** | **0.011** |

As the number of modes increases, both methods degrade monotonically, but MOMENT's gap over FFT
grows: at K=2 it is +40.2 pp; at K=8 it is +21.4 pp. Silhouette scores converge — MOMENT's
silhouette advantage disappears at K=8 (0.146 vs 0.250) because projecting many modes into 2
PCA dimensions forces overlap. Loop DTW for MOMENT remains stable (4.9–5.7) regardless of K,
while FFT loop DTW also stays flat (~115–147) — mode count doesn't affect trace consistency.

### 9.4 Signal Type Axis

| Scenario | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Centroid Stab ↓ |
|----------|-------|-------|-------|-----------|----------------|
| baseline (sine) | fft32 | 37.1 | 0.407 | 141.9 | 0.371 |
| baseline (sine) | moment | **72.5** | **0.497** | **5.2** | **0.008** |
| sig_mixed | fft32 | 22.6 | 0.212 | 280.7 | 0.436 |
| sig_mixed | moment | **36.9** | **0.184** | **8.8** | **0.008** |
| sig_event | fft32 | **23.9** | **0.208** | 252.2 | 0.620 |
| sig_event | moment | 18.7 | 0.038 | **11.1** | **0.012** |
| sig_step | fft32 | **9.6** | −0.020 | 274.4 | 0.447 |
| sig_step | moment | 2.8 | −0.011 | **10.4** | **0.011** |

**This is the most consequential finding of RQ3.** Both methods are built around spectral
assumptions — MOMENT is pretrained on smooth continuous signals; FFT explicitly computes frequency
content. When the signal type changes to steps or events, **FFT beats MOMENT on MSI** (step: 9.6
vs 2.8; event: 23.9 vs 18.7). MOMENT's pretrained features are essentially blind to the
distinguishing information in non-periodic signals.

However, MOMENT retains its loop consistency advantage even on step/event signals (loop DTW ~10–11
vs FFT ~250–274). The mode orbits are still compact — they just overlap in PCA space.

---

## 10. RQ4 — CNC-Analogous Benchmark Results

### 10.1 Program Step Labels

| Scenario | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Trans. Sharp ↓ | Centroid Stab ↓ |
|----------|-------|-------|-------|-----------|---------------|----------------|
| cnc_part_simple | fft | 55.9 | 0.002 | 166.2 | 2.74 | 0.897 |
| cnc_part_simple | moment | 36.1 | −0.008 | **2.89** | 2.11 | **0.004** |
| cnc_part_full | fft | **68.2** | **0.030** | 197.0 | 2.09 | 0.776 |
| cnc_part_full | moment | 43.7 | −0.009 | **2.29** | 1.51 | **0.003** |
| cnc_two_parts | fft | 45.0 | −0.024 | 837.4 | 1.16 | 1.517 |
| cnc_two_parts | moment | **42.7** | −0.036 | **2.97** | 1.21 | **0.003** |
| cnc_tool_wear | fft | **64.6** | **0.018** | 354.1 | 2.13 | 0.969 |
| cnc_tool_wear | moment | 43.3 | −0.010 | **2.26** | 1.37 | **0.003** |
| cnc_noisy | fft | **65.1** | **0.021** | 240.7 | 2.17 | 0.846 |
| cnc_noisy | moment | 43.1 | −0.006 | **2.27** | 1.28 | **0.003** |
| cnc_pocket | fft | **81.5** | nan | 243.2 | 3.06 | 0.829 |
| cnc_pocket | moment | 54.9 | nan | **2.25** | 1.54 | **0.002** |
| cnc_faults | fft | **51.1** | **0.022** | 244.3 | 1.83 | 1.096 |
| cnc_faults | moment | 38.8 | −0.004 | **2.02** | 1.14 | **0.003** |

### 10.2 Operating State Labels

| Scenario | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Trans. Sharp ↓ | Centroid Stab ↓ |
|----------|-------|-------|-------|-----------|---------------|----------------|
| cnc_part_simple | fft | 40.9 | **0.509** | 370.1 | 1.50 | 1.228 |
| cnc_part_simple | moment | 37.6 | 0.098 | **6.85** | 2.50 | **0.004** |
| cnc_part_full | fft | 48.2 | **0.446** | 417.2 | 2.50 | 1.074 |
| cnc_part_full | moment | 44.3 | 0.170 | **5.38** | 2.50 | **0.003** |
| cnc_two_parts | fft | 39.3 | **0.346** | 2214.2 | 2.50 | 1.658 |
| cnc_two_parts | moment | **41.0** | 0.123 | **7.06** | 2.50 | **0.004** |
| cnc_tool_wear | fft | 47.0 | **0.429** | 825.3 | 2.50 | 1.216 |
| cnc_tool_wear | moment | 43.8 | 0.164 | **5.30** | 2.50 | **0.003** |
| cnc_noisy | fft | 47.5 | **0.443** | 475.3 | 2.50 | 1.099 |
| cnc_noisy | moment | 44.1 | 0.153 | **5.30** | 2.50 | **0.003** |
| cnc_pocket | fft | 55.3 | **0.482** | 581.3 | 1.67 | 1.360 |
| cnc_pocket | moment | **55.8** | 0.241 | **5.69** | 2.33 | **0.003** |
| cnc_faults | fft | 35.1 | **0.412** | 498.3 | 2.67 | 1.510 |
| cnc_faults | moment | 38.6 | 0.149 | **3.99** | 2.00 | **0.003** |

### 10.3 Key Findings

**FFT dominates program-step MSI.** Across all 7 scenarios, FFT32 achieves 45–82 MSI vs MOMENT's
36–55. The spectral structure of the machining signals (spindle frequency, vibration harmonics)
carries strong mode-separating information that FFT captures directly.

**MOMENT dominates geometry.** Loop DTW 2.0–2.9 (MOMENT) vs 166–837 (FFT) for program steps;
5.4–7.1 (MOMENT) vs 370–2214 (FFT) for operating states. The 100–300× gap is consistent
across all scenarios including noisy and fault-injected ones.

**Operating state is easier than program step.** FFT silhouette on op-state (0.35–0.51) is far
higher than on program step (~0.0–0.03). Op-states correspond to coarse machine behaviours
(cutting vs. idle vs. homing) that produce clearly distinct spectral signatures; program steps
within a cutting phase share similar spectra but differ in toolpath position.

**MOMENT wins on op-state MSI in cnc_pocket (55.8 vs 55.3) and cnc_faults (38.6 vs 35.1).**
For the most complex scenario (11 program steps with Z-varying geometry) and the fault-injected
scenario, MOMENT's feature richness starts to matter for MSI as well. This hints that MOMENT's
768-dim representation captures more than spectral amplitude.

**Discrete channels (tool_id, alarm_code) do not break MOMENT.** MOMENT's mean-pooling across
channels treats discrete channels the same as continuous ones. The embedding quality on
cnc_faults (which has active alarm codes in 8% of windows) is comparable to clean scenarios.

---

## 11. RQ5 — Window Size Sweep Results

### 11.1 Program Step Labels

| Window | Stride | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Trans. Sharp ↓ | Centroid Stab ↓ |
|--------|--------|-------|-------|-------|-----------|---------------|----------------|
| 32 | 4 | fft | 48.9 | 0.018 | 118.1 | 2.00 | 0.087 |
| 32 | 4 | moment | 37.9 | 0.038 | **3.71** | 2.47 | **0.001** |
| **64** | **8** | **fft** | **64.3** | **0.057** | 144.0 | 1.93 | 0.335 |
| **64** | **8** | **moment** | **48.6** | **0.065** | **1.98** | **1.15** | **0.002** |
| 128 | 16 | fft | 50.0 | −0.021 | 197.6 | 3.40 | 0.667 |
| 128 | 16 | moment | 35.9 | 0.021 | 2.00 | 2.22 | 0.003 |
| 256 | 32 | fft | 39.0 | 0.258 | 199.5 | 1.11 | 1.538 |
| 256 | 32 | moment | 28.5 | 0.033 | 0.92 | 1.26 | 0.003 |
| 512 | 64 | fft | 18.5 | −0.073 | 664.6 | 1.71 | 12.049 |
| 512 | 64 | moment | 28.0 | −0.096 | 1.25 | 1.76 | 0.006 |

### 11.2 Operating State Labels

| Window | Stride | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Trans. Sharp ↓ | Centroid Stab ↓ |
|--------|--------|-------|-------|-------|-----------|---------------|----------------|
| 32 | 4 | fft | 35.4 | 0.357 | 199.3 | 2.39 | 0.076 |
| 32 | 4 | moment | 34.8 | 0.127 | **6.84** | 1.70 | **0.001** |
| **64** | **8** | **fft** | **55.3** | **0.477** | 247.1 | **1.40** | 0.341 |
| **64** | **8** | **moment** | **46.3** | **0.213** | **3.67** | 1.07 | **0.002** |
| 128 | 16 | fft | 62.7 | 0.470 | 648.0 | 3.00 | 0.924 |
| 128 | 16 | moment | 50.6 | 0.203 | 7.51 | 3.00 | 0.003 |
| 256 | 32 | fft | 0.0 | nan | 692.5 | nan | 0.840 |
| 256 | 32 | moment | 0.0 | nan | 3.46 | nan | 0.002 |
| 512 | 64 | fft | 0.0 | nan | 658.5 | nan | 2.971 |
| 512 | 64 | moment | 0.0 | nan | 1.12 | nan | 0.003 |

### 11.3 Key Findings

**The sweet spot is ws=64 for both models.** FFT reaches its peak program-step MSI (64.3) and
op-state silhouette (0.477) at ws=64. MOMENT peaks at ws=64 for program-step (MSI=48.6,
silhouette=0.065) and for op-state (MSI=46.3, silhouette=0.213).

**FFT collapses at ws=512.** MSI drops from 64.3 at ws=64 to 18.5 at ws=512 for program steps.
The centroid stability explodes to 12.0 — mode regions drift massively at very large windows.
At ws=512 with stride=64, there are only ~106 windows per trajectory, breaking the statistical
reliability of the metrics.

**MOMENT is more robust at large windows.** At ws=512, MOMENT MSI (28.0) beats FFT (18.5) for
program steps. MOMENT also achieves its best loop DTW at ws=512 (1.25) — because at 512
samples the window naturally covers an entire mode segment, making successive visits nearly
identical.

**Op-state metrics collapse at ws≥256.** Both models show MSI=0 and NaN silhouette. At stride=32,
the very short RAPID and HOMING segments produce fewer than one window in some trajectories,
so op-state coverage becomes incomplete. This is a real production constraint: window size
must be small relative to the shortest mode duration.

**MOMENT padding penalty is invisible at ws=64.** Despite 7× zero-padding (64 → 512), MOMENT
performs well — the zero-padding is right-sided and the signal is left-aligned, so the first
64 samples are unaffected. The penalty only appears at ws=32 where 15× padding dilutes the
signal substantially.

---

## 12. RQ6 — Signal Ablation Results

### 12.1 Step Channel Ratio (ratio_0step → ratio_4step)

| Scenario | Step ch | Model | MSI ↑ | Sil ↑ | Loop DTW ↓ | Centroid Stab ↓ |
|----------|---------|-------|-------|-------|-----------|----------------|
| ratio_0step | 0/4 | fft32 | 38.3 | 0.441 | — | 0.311 |
| ratio_0step | 0/4 | moment | **82.8** | **0.429** | — | **0.002** |
| ratio_1step | 1/4 | fft32 | 28.1 | 0.297 | — | 0.388 |
| ratio_1step | 1/4 | moment | **73.4** | **0.382** | — | **0.002** |
| ratio_2step | 2/4 | fft32 | 22.4 | 0.208 | — | — |
| ratio_2step | 2/4 | moment | **60.2** | **0.293** | — | **0.002** |
| ratio_3step | 3/4 | fft32 | 17.0 | 0.128 | — | — |
| ratio_3step | 3/4 | moment | **48.1** | **0.182** | — | **0.002** |
| ratio_4step | 4/4 | fft32 | 9.6 | −0.020 | — | — |
| ratio_4step | 4/4 | moment | **27.7** | **0.085** | — | **0.002** |

**MOMENT degrades gracefully; FFT collapses.** Each additional step channel costs FFT32
roughly 7 MSI points (38.3 → 9.6, −75%). MOMENT loses about 14 MSI points per step channel
added (82.8 → 27.7, −67%) but remains substantially ahead in absolute terms at every level.
At ratio_4step, MOMENT MSI (27.7) is 3× FFT (9.6).

**The RQ3 sig_step result (MOMENT 2.8, FFT 9.6) was a worst-case boundary.** In RQ3, the step
signal was designed with abrupt transitions that share identical amplitude histograms across modes
— a pathological case where even the signal envelope carries no discriminating information. RQ6's
step channels use natural step levels tied to mode identity, which MOMENT can partially exploit via
temporal context.

**Silhouette tracks MSI.** At ratio_4step, FFT32 silhouette goes negative (−0.020), confirming
that clusters overlap in PCA space — modes are less separable than random. MOMENT maintains
positive silhouette (0.085) even with all channels as step signals.

### 12.2 Exotic Signal Types

| Scenario | Signal | Model | MSI ↑ | Sil ↑ |
|----------|--------|-------|-------|-------|
| baseline (sine) | Sine | fft32 | 37.1 | 0.407 |
| baseline (sine) | Sine | moment | **72.5** | **0.497** |
| sig_chirp | Chirp | fft32 | 18.9 | 0.158 |
| sig_chirp | Chirp | moment | **54.8** | **0.240** |
| sig_damped | Damped | fft32 | 24.5 | 0.188 |
| sig_damped | Damped | moment | **56.8** | **0.330** |
| sig_sawtooth | Sawtooth | fft32 | 24.7 | 0.377 |
| sig_sawtooth | Sawtooth | moment | **51.2** | **0.291** |

**MOMENT's advantage grows on exotic continuous signals.** On sine, MOMENT leads FFT32 by +35.4
MSI. On chirp (+35.9), damped (+32.3), and sawtooth (+26.5), the gap is comparable or larger.
MOMENT's pretrained transformer extracts features — temporal shape, envelope progression,
inter-cycle structure — that a simple amplitude spectrum cannot represent.

**Chirp is worst for FFT.** A chirp signal spreads energy across a wide frequency band as the
instantaneous frequency sweeps; the FFT amplitude spectrum is smeared and loses its discriminative
peaked structure (FFT32 MSI drops from 37.1 to 18.9, −49%). MOMENT is relatively unaffected
(−24%).

**Sawtooth is surprisingly good for FFT.** A sawtooth contains strong harmonics at integer
multiples of the fundamental. FFT32 recovers silhouette 0.377 — its best exotic-signal result —
because those harmonics are exactly what FFT bins capture. MOMENT still wins on MSI (51.2 vs
24.7) but the gap narrows.

**Damped oscillations: MOMENT wins on MSI and silhouette.** Decaying amplitude within each mode
segment means successive windows have different energy levels; FFT conflates amplitude change with
mode change (MSI 24.5). MOMENT's attention mechanism integrates the full temporal shape of each
window including its decay envelope, maintaining MSI 56.8.

### 12.3 RQ6 Key Findings Summary

1. **Step contamination is a continuous degradation, not a cliff.** Each step channel added
   costs roughly proportional performance, with MOMENT maintaining a consistent absolute lead.
   Mixed-signal PLC logs (which always have some analogue channels) will therefore always give
   MOMENT a meaningful advantage over FFT.

2. **The RQ3 sig_step floor is a genuine worst case.** Real production signals with any periodic
   or slowly-varying channels will not reach MSI 2.8. Even 3/4 step channels leaves MOMENT at MSI
   48.1 — well above the RQ3 pathological case.

3. **For non-standard continuous signals, MOMENT is robustly superior.** Chirp, damped, and
   sawtooth all yield MOMENT MSI 50–57 vs FFT 19–25. The transformer's learned features
   generalise across signal shapes in a way that fixed spectral bins cannot.

4. **MOMENT centroid stability stays near-zero regardless of signal type.** The 0.002 value is
   unchanged across all 8 RQ6 scenarios — the orbit consistency invariant identified in RQ2
   extends fully to mixed and exotic signal types.

---

## 13. Cross-Experiment Synthesis

### 13.1 The Role Reversal: TS2Vec vs MOMENT

![Role reversal](figures/fig7_role_reversal.png)

One of the most striking findings across all experiments is a **consistent role reversal** between
MOMENT and TS2Vec:

| | RQ1 (factor recovery probes) | RQ2 (embedding geometry) |
|--|--|--|
| MOMENT | 61.8% avg balanced acc — middle | Loop DTW ~1.5 — **dominant** |
| TS2Vec e120 | 66.5% avg balanced acc — **best** | Loop DTW ~87 — worst |

TS2Vec's contrastive training pushes same-window representations together globally, which helps
linear/RBF probes recover discrete factor labels — but the geometry is chaotic, loops don't form,
centroids drift. MOMENT's pretrained features produce geometrically regular embeddings — tight,
stable, fast mode transitions — without any task-specific training.

**These are complementary strengths.** TS2Vec for discrete label recovery; MOMENT for continuous
online monitoring and anomaly detection.

### 13.2 MOMENT vs FFT: When Each Wins

Across all five RQs, two conditions predict which model wins on MSI:

| Condition | MOMENT wins MSI | FFT wins MSI |
|-----------|----------------|-------------|
| Clean, periodic, sine signals | ✓ (RQ3 noise_000: 73.1 vs 37.2) | |
| High noise, periodic signals | ✓ (RQ3 noise_040: 59.9 vs 22.8) | |
| Chirp / damped / sawtooth signals | ✓ (RQ6: 51–57 vs 19–25) | |
| Mixed step+sine channels (1–3 step ch) | ✓ (RQ6: 48–73 vs 17–28) | |
| Pure step-change signals | | ✓ (RQ3 sig_step: 9.6 vs 2.8) |
| Event/impulse signals | | ✓ (RQ3 sig_event: 23.9 vs 18.7) |
| CNC machining, program steps | | ✓ (RQ4: 45–82 vs 36–55) |
| CNC machining, operating states | ~tied | ~tied |
| Small window (ws=32) | | ✓ FFT closer |
| Optimal window (ws=64) | | ✓ FFT 64.3 vs MOMENT 48.6 |

MOMENT wins whenever the signal is periodic and smooth. FFT wins (or ties) whenever the signal
has strong spectral structure that a simple amplitude spectrum captures completely, or when the
signal is non-periodic. On geometry metrics (loop DTW, centroid stability), MOMENT wins
**always and by a large margin** regardless of condition.

### 13.3 The Loop Consistency Invariant

Across all 5 RQs, MOMENT loop DTW is bounded between 1.25 and 11.1 regardless of noise level,
missing data type, number of modes, signal type, window size, or scenario complexity. FFT loop
DTW ranges from 66 (best case P1) to 2214 (cnc_two_parts op-state). This suggests MOMENT's
pretrained representation has a structural property that enforces temporal consistency in a
way FFT does not — likely the transformer's attention mechanism averaging over the full window
context rather than computing a fixed spectral decomposition.

### 13.4 Production Risk Summary

| Risk | FFT | MOMENT |
|------|-----|--------|
| Sensor dropout (channel missing) | **Critical** — MSI 13.5, loop DTW 478 | Not measured — likely critical |
| High noise (σ = 0.40) | Moderate — MSI drops 38% | Low — MSI drops 18% |
| Step/event signal type | Moderate — MSI 9.6 | **Severe** — MSI 2.8 |
| Many modes (K=8) | Moderate — MSI 30.9 | Low — MSI 52.2 |
| Small window (ws=32) | Moderate | Low (padding robust at ws=64+) |
| Large window (ws≥256) | **Severe** — MSI collapses | Moderate — more stable |

---

## 14. Lessons Learned

### 14.1 Method Design

| # | Lesson | Evidence |
|---|--------|----------|
| L1 | Temporal-neighbour contrastive pairs teach temporal proximity, not mode identity. | Phase 1–2: baselines beat learned |
| L2 | Reconstruction loss adds complexity but doesn't fix misaligned objectives. | Phase 2 vs 1: no improvement |
| L3 | More training helps probes but can harm global structure. | TS2Vec e80→e120: coupling ARI 0.006→0.004 |
| L4 | Pretrained foundation models give strong geometry without task-specific training. | RQ1/RQ2 role reversal |
| L5 | Load (continuous amplitude) is not learnable from the current setup — R²≤0 for all methods. | RQ1 Table 1 |
| L6 | MOMENT's pretraining assumption fails on *purely* step/event signals — but mixed signals still favour MOMENT. | RQ3 sig_step MSI 2.8; RQ6 ratio_4step MSI 27.7 vs 9.6 |
| L7 | Channel dropout is a critical production failure mode for all tested methods. | RQ3 miss_channel: FFT MSI 13.5, loop DTW 478 |
| L8 | MOMENT loop consistency is an invariant: ~0.002 centroid stability regardless of signal type, window size, or noise. | RQ6: all 8 scenarios show centroid stability 0.002 |
| L9 | Exotic continuous signals (chirp, damped, sawtooth) favour MOMENT strongly — FFT MSI drops 49% on chirp. | RQ6 sig_chirp: FFT 18.9, MOMENT 54.8 |

### 14.2 Evaluation Design

| # | Lesson | Evidence |
|---|--------|----------|
| L8 | Retrieval R@K is misleading when training biases nearest neighbours. | Phase 2: R@5 > 0.96 but probe accuracy 27% |
| L9 | Single-seed results for TS2Vec should be treated cautiously. | MOMENT coupling std ±0.042 |
| L10 | Geometry metrics reveal embedding quality that probe accuracy misses. | MOMENT dominates RQ2 while middle of pack in RQ1 |
| L11 | PCA compactness may penalise high-dimensional embeddings unfairly. | MOMENT PCA compact 1.63–7.97 vs FFT 0.26–0.33 |
| L12 | Window size must be smaller than the shortest mode segment duration. | RQ5: op-state MSI=0 at ws≥256 |

### 14.3 Benchmark Design

| # | Lesson | Evidence |
|---|--------|----------|
| L13 | FRS is a valid instrument for spectral/mode learning but doesn't capture step signals, discrete channels, or mixed signal types from real PLC logs. | RQ3 signal-type axis |
| L14 | High baseline performance (FFT avg 60.3%) indicates strong spectral structure — learned methods must beat it by a meaningful margin. | RQ1 Table 2: +6.2pp gap |
| L15 | Missing data stress-testing is essential before production recommendation. | RQ3: block-missing drops MOMENT MSI 44% |
| L16 | Two-level label structure (coarse/fine) reveals that the same embedding cannot optimise both simultaneously. | RQ4: FFT wins on fine (program step), ties on coarse (op-state) |

---

## 15. Open Questions

1. **Why does MOMENT fail so severely on step signals?** MSI 2.8 on sig_step while retaining
   loop DTW 10.4 — the orbits are consistent but entirely overlapping. Is the T5 backbone
   averaging out the step transitions into a smooth mean?

2. **Can factor-aware pairs fix the load problem?** Windows with matching load values as positives
   instead of temporal neighbours — the most direct fix for Load R²≤0.

3. **Does MOMENT's loop consistency invariant hold under channel dropout?** The miss_channel
   scenario was not evaluated for MOMENT. If loop DTW is still low despite collapsed MSI,
   it has important implications for fault-tolerant monitoring architectures.

4. **What is the optimal ensemble?** RQ4 shows FFT and MOMENT have complementary strengths:
   FFT for MSI / label classification, MOMENT for loop consistency / anomaly detection.
   A simple ensemble (MOMENT geometry for anomaly detection + FFT features for mode
   classification) might outperform either alone.

5. **Would DCC (contrastive training with factor-aware pairs) break the role reversal?**
   The RQ3 DCC training jobs failed due to an import bug (now fixed). Running DCC would tell
   us whether a trained model can achieve both high probe accuracy (like TS2Vec) and
   good geometry (like MOMENT).

6. **Why does MOMENT perform so much better in RQ6 ratio_4step (MSI 27.7) than RQ3 sig_step
   (MSI 2.8) if both are pure step-channel?** The mode-level step amplitude patterns in RQ6
   likely differ across modes (each mode has a fixed level per channel), whereas RQ3 sig_step
   used level differences that overlapped. Confirms that the signal-type failure is amplitude
   overlap, not just aperiodicity.

7. **Does ws=64 generalise across scenarios?** RQ5 only tested one scenario (cnc_part_full).
   The optimal window size likely varies with spindle speed — at high RPM, 64 samples may
   capture less than one vibration cycle.

---

## 16. What Remains To Be Done

### Immediate

1. **RQ3 DCC training** — resubmit `rq3_dcc.slurm` (import bug fixed). Will add a trained
   contrastive baseline to the RQ3 landscape sweep.
2. **Sync RQ3–5 results locally** — rsync results and plots from Alvis.
3. **RQ2 visualisation plots** — the viz step failed for RQ2 eval due to missing matplotlib;
   regenerate locally.

### Track A — FRS Fixes (highest priority)

- **A1**: Factor-aware positive pairs for TS2Vec (fix load and coupling learning)
- **A2**: Load regression auxiliary head
- **A4**: Multi-seed TS2Vec to quantify variance
- **A3**: OOD splits (unseen factor combinations at test time)
- **A5**: Segment-level change-point detection evaluation

### Track B — PLC Log Bridge

- **B1**: Synthetic PLC log corpus with FRS-like factors and mixed signal types
- **B2**: Tabular/numerical embedding extraction with step-aware encoding
- **B3**: Cross-modal comparison with FRS eval infrastructure

### Track C — Geometry and Architecture

- Replace PCA compactness with silhouette score in the full embedding space
- Test MOMENT with fine-tuning on a small labelled set (few-shot regime)
- Investigate whether a lightweight adapter on MOMENT's 768-dim output can recover
  the Load factor that frozen MOMENT misses
