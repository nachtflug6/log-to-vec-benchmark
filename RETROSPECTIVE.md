# Project Retrospective

Last updated: 2026-04-29

This document synthesises everything learned so far: what was tried, what the numbers showed, what
failed, what is still open, and what should not be repeated. It is the primary reference for
planning next experiments.

---

## Original Goal

Learn useful embeddings for **PLC-type log files**: timestamped records of sensor readings and
discrete machine state codes from industrial control systems. No free-text messages, no JSON
payloads. The log format is essentially a serialised multivariate time series: each row is a
timestamped snapshot of sensor channels and integer state codes (STOPPED=0, IDLE=1, RUNNING=2,
FAULT=3, …). The motivating question was whether a learned representation could capture system
behaviour—operating regime, anomaly, mode change—without hand-labelled data.

Scope: the target domain is **PLC / SCADA logs**, not general syslog or application logs.
This distinction matters for method selection: LLM-based text embedding is largely irrelevant
(there is no meaningful text), while the parse-to-time-series approach is a natural fit.

---

## The Drift: PLC Logs → Pure Synthetic Time Series

The team moved from the original PLC log framing to a pure synthetic benchmark (FRS). For the
target domain this is a **less severe** drift than it first appears, because PLC logs are already
numeric: sensor values are floats, machine states are integer codes. Parsing a PLC log to a
multivariate time series matrix is straightforward—pivot by channel, resample to a fixed grid,
encode state codes as additional integer channels.

What the FRS benchmark does **not** capture from real PLC logs:
- **Named channels**: real logs have meaningful channel names (e.g. `Motor_Speed`, `Tank_Pressure`)
  that encode domain knowledge. FRS channels are anonymous.
- **Irregular per-channel sampling**: different sensors may log at different rates. FRS uses a
  uniform sample rate.
- **Discrete state codes mixed with continuous readings**: a window contains both float sensor
  data and integer state codes; FRS treats all channels as continuous.
- **Alarm and fault events**: PLC systems produce sparse alarm records alongside continuous
  readings; FRS does not model sparsity.

These are engineering gaps, not fundamental mismatches. The FRS benchmark is a valid scientific
instrument for the core learning problem; the bridge to real PLC data is tractable.

**Decision for next experiments**: Three parallel tracks are planned (see
[docs/next_experiments.md](docs/next_experiments.md)):
- Track A: Fix the FRS/time-series pipeline (immediate, most actionable).
- Track B: PLC log parsing bridge—convert real or synthetic PLC logs to the FRS evaluation
  infrastructure (natural fit given the numeric-heavy format).
- Track C: LLM-based embedding—low priority for PLC logs (minimal text), but worth a brief
  comparison if state transition labels or alarm text are available.

---

## Timeline of Phases

### Phase 1 — Contrastive LSTM on Toy Logs (contrastive-phase1)

Code: [archive/legacy/contrastive_phase1/](archive/legacy/contrastive_phase1/)

**What was tried:**
- LSTM encoder trained with NT-Xent contrastive loss.
- Positive pairs defined as temporal neighbours within a trajectory.
- Input: toy CSV logs generated synthetically (`data/toy_logs.csv`).
- Sine-wave generators used as simple synthetic data (`examples/generate_sine_logs.py`).
- Evaluation: basic clustering and retrieval tests, no formal probe suite.

**Results:** No formal numeric results were recorded. The approach was discontinued because:
- Temporal-neighbour positives teach the model "what is nearby in time", not "what shares a mode".
- No ground-truth factorised labels existed for the toy data.
- Evaluation was ad-hoc (visualisations only).

**Lesson:** Temporal-neighbour contrastive learning on unstructured toy logs cannot be evaluated
rigorously. A controlled synthetic benchmark with known latent structure is necessary before
claiming anything about learned representations.

---

### Phase 2 — TCN Hybrid on FSSS (version2 branch)

Code: [src/version2/](src/version2/), [examples/fsss/](examples/fsss/)
Artifacts: [outputs/version2/](outputs/version2/)

**What was tried:**
- TCN (Temporal Convolutional Network) encoder with dilated residual conv blocks.
- Hybrid contrastive + reconstruction loss (`src/version2/training/hybrid_losses.py`).
- Augmentation: Gaussian noise, time shifts, span masking for reconstruction objective.
- Datasets: simpler FSSS (Factorised State-Space Sequence) datasets—earlier, smaller variants of FRS.
- Evaluation suite: linear and RBF probes, retrieval, clustering, transition analysis
  (`src/version2/evaluation/eval_v2.py`).
- Baselines: raw statistics, FFT, PCA (`src/version2/evaluation/baseline_features.py`).

**Key results (from outputs/version2/reports/run_003/ and compare/):**

Probe balanced accuracy on the FSSS test set:

| Factor      | Baseline (best) | Learned (best) | Delta |
|-------------|-----------------|----------------|-------|
| mode_id     | 0.277 (summary) | 0.267 (RBF)    | **−0.010** — baseline wins |
| spectral_id | 0.701 (summary) | 0.653 (linear) | **−0.048** — baseline wins |
| coupling_id | 0.413 (summary) | 0.423 (linear) | +0.010 — marginal |
| transition  | 0.558 (FFT)     | 0.534 (RBF)    | **−0.024** — baseline wins |
| mean_load   | ≈0 (all)        | ≈0 (all)       | tied at zero |

Retrieval R@5 (from full_report.md):
- mode_id: 0.961 — high but misleading (temporal-neighbour bias)
- spectral_id: 0.983
- coupling_id: 0.993

Clustering ARI (from full_report.md):
- mode_id: 0.134 — very poor global structure
- spectral_id: 0.030
- coupling_id: 0.020

**Interpretation:** The learned TCN embedding does not beat hand-crafted FFT/summary baselines
on any factor except coupling by a tiny margin. High retrieval is an artefact of the training
objective, not semantic understanding. Global structure (clustering) is very poor.

**Note on eval_v2 dataset:** An earlier eval_v2 run showed 94.7% mode probe and R²=1.0 for load.
These numbers are **invalid**—that dataset had a degenerate split where coupling and transition had
only one class in train, and load leaked perfectly. Do not cite those numbers.

**Lesson:** TCN with temporal-neighbour contrastive loss + reconstruction does not extract
factorised structure. The training objective is misaligned with the evaluation goal. Baselines
are formidable.

---

### Phase 3 — Formal RQ1 Benchmark (experiment1/rq1_recoverability branch)

Code: [experiments/rq1_recoverability/](experiments/rq1_recoverability/),
[src/version2/](src/version2/), [src/moment/](src/moment/)
Artifacts: [experiments/rq1_recoverability/artifacts/](experiments/rq1_recoverability/artifacts/),
[experiments/rq1_recoverability/reports/](experiments/rq1_recoverability/reports/)

**What was tried:**
- Formal FRS (Factorised Regime Sequence) generator with reproducible profiles
  (`experiments/rq1_recoverability/src/rq1/generation/factorized_regime_sequence_generator.py`).
- Two primary datasets: `frs_clean_vnext_long` and `frs_noisy_vnext_long`.
- Trajectory-level splits (70/15/15) with leakage checking.
- Three method families evaluated:
  - **MOMENT pretrained** (foundation model, 3 seeds: 42/53/64)
  - **TS2Vec-style contrastive** (trained from scratch, 1 seed, e80 and tuned e120)
  - **Hand-crafted baselines**: FFT, Summary statistics, Raw flatten
- Numbered pipeline: scripts 01–13 under `experiments/rq1_recoverability/scripts/`.
- Alvis HPC cluster integration for reproducible execution.

**Key results — Table 1 (RBF probe balanced accuracy):**

| Method           | Seeds | mode   | spectral | coupling | transition | load R² |
|------------------|-------|--------|----------|----------|------------|---------|
| MOMENT           | 3     | 0.466 ±0.012 | 0.756 ±0.018 | 0.633 ±0.042 | 0.576 ±0.042 | −0.037 ±0.049 |
| TS2Vec e80       | 1     | 0.398  | 0.694    | 0.625    | 0.527      | −0.006  |
| TS2Vec e120      | 1     | **0.514** | **0.767** | **0.713** | 0.527 | −0.006  |
| Baseline FFT     | 1     | 0.439  | 0.696    | 0.673    | —          | −0.006  |
| Baseline Summary | 1     | 0.431  | 0.653    | 0.634    | —          | −0.006  |

Dataset: frs_clean_vnext_long. Noisy results are 3–7% lower across all factors for all methods.

**Key results — Table 2 (Retrieval precision@10):**

| Method      | mode  | spectral | coupling | avg   |
|-------------|-------|----------|----------|-------|
| MOMENT      | 0.340 | 0.592    | 0.526    | 0.486 |
| TS2Vec e80  | 0.378 | 0.593    | 0.551    | 0.507 |
| TS2Vec e120 | **0.407** | **0.606** | **0.583** | **0.532** |

**Key results — Table 3 (Clustering ARI):**

| Method      | mode  | spectral | coupling | avg   |
|-------------|-------|----------|----------|-------|
| MOMENT      | 0.151 | 0.180    | 0.058    | 0.130 |
| TS2Vec e80  | 0.153 | **0.207** | 0.006   | 0.122 |
| TS2Vec e120 | **0.157** | 0.194 | 0.004   | 0.118 |

All clustering ARI values are very low. More training (e80→e120) does not improve clustering
and actually degrades coupling ARI.

**Key results — Table 4 (Baseline vs learned, avg factor balanced accuracy):**

| Method           | Clean avg | Noisy avg |
|------------------|-----------|-----------|
| Baseline FFT     | 0.603     | 0.576     |
| Baseline Summary | 0.573     | 0.579     |
| MOMENT           | 0.618     | 0.590     |
| TS2Vec e80       | 0.572     | 0.575     |
| TS2Vec e120      | **0.665** | **0.607** |

The gap between the best learned method (TS2Vec e120, 66.5%) and the best baseline (FFT, 60.3%)
is only **6.2 percentage points** on clean data. On noisy data the gap narrows further.

**Summary findings:**
- Load factor R² ≤ 0 for **every method** in **every setting**. Load is completely unlearned.
- Spectral structure is easiest to learn (strongest signal, easy for FFT baseline too).
- Mode classification is marginal; model barely beats FFT on clean data.
- Global clustering is uniformly poor (ARI < 0.21); local retrieval is high but misleading.
- Noise degrades all methods; MOMENT degrades least on spectral; TS2Vec degrades least on load.
- More training epochs (e80→e120) helps probes but not clustering.
- Multi-seed variance (MOMENT only): small for spectral (±0.018), larger for coupling (±0.042).

---

## Open Problems

These problems are confirmed by both empirical results and analysis in
[docs/research_findings.md](docs/research_findings.md) and
[context/output_1.txt.txt](context/output_1.txt.txt).

| # | Problem | Evidence |
|---|---------|----------|
| 1 | **Training–eval mismatch**: temporal-neighbour objective ≠ factor recovery | Phase 2 baseline beats learned on mode/spectral/transition |
| 2 | **Positive pairs misaligned**: temporal neighbours ≠ same-factor windows | High retrieval despite low probe accuracy |
| 3 | **Load not learned**: R² ≤ 0 across all methods and settings | Tables 1, 4 |
| 4 | **Poor global structure**: clustering ARI < 0.21 everywhere | Table 3 |
| 5 | **Retrieval bias**: high retrieval reflects training objective, not semantics | R@5 > 0.96 in Phase 2 despite 24% mode probe |
| 6 | **Thin margin over baselines**: best learned method beats FFT by 6.2pp clean | Table 4 |
| 7 | **No OOD evaluation**: all factor combos seen during training | Benchmark design gap |
| 8 | **Single-seed instability**: most runs at 1 seed | Only MOMENT has 3 seeds |
| 9 | **Transition smearing**: temporal smoothing blurs transition windows | Transition probe ≤ spectral probe |
| 10 | **Spectral dominance**: frequency signal may shortcut learning | FFT baseline strong; load absent |
| 11 | **PLC-specific gaps**: named channels, mixed discrete/continuous channels, irregular sampling not in FRS | Engineering gap |

---

## What NOT to Retry

- **Temporal-neighbour positive pairs without change**: tried in Phases 1 and 2, failed to produce
  factor-level global structure. Will produce the same result.
- **Larger TCN with same objective**: more capacity under the same loss does not fix misalignment.
- **Retrieval-only evaluation**: high R@5 was shown to be misleading in both Phase 2 and Phase 3.
- **Degenerate/small FSSS splits**: the earlier FSSS datasets had split issues (single class in
  train for some factors). Always use `frs_*_vnext_long` profiles.
- **Reporting R²=1.0 for load**: that result came from a degenerate dataset; discard it.

---

## What Worked Marginally and Is Worth Building On

- **FRS benchmark infrastructure**: the formal pipeline (scripts 01–13, trajectory-level splits,
  registry, leakage checks) is solid. Keep it as the evaluation platform.
- **TS2Vec-style encoder with tuning**: e120 is the best learned method so far. Worth using as a
  baseline for new objectives, not as the primary method under study.
- **Multi-view evaluation**: probes + retrieval + clustering + transition analysis together give
  a diagnostic picture. This should remain standard.
- **MOMENT pretrained embeddings**: competitive, stable across seeds, useful as an upper-bound
  reference for what a pretrained foundation model achieves without any task-specific training.
- **Alvis HPC integration**: smoke-deploy pipeline verified 2026-04-29; ready for scale.

---

## Cross-reference

- Research questions: [docs/research_questions.md](docs/research_questions.md)
- Benchmark design: [docs/benchmark_design.md](docs/benchmark_design.md)
- Detailed findings: [docs/research_findings.md](docs/research_findings.md)
- Next experiments: [docs/next_experiments.md](docs/next_experiments.md)
- Phase reports: [experiments/history/](experiments/history/)
- Result tables: [experiments/rq1_recoverability/reports/result_tables/](experiments/rq1_recoverability/reports/result_tables/)
- Context PDFs and analysis: [context/](context/) (untracked; local only)
