# Next Experiments

Last updated: 2026-04-29
Prerequisite reading: [RETROSPECTIVE.md](../RETROSPECTIVE.md)

This document describes the prioritised roadmap for the next phase of experiments.

**Target domain**: PLC / SCADA log files — timestamped sensor readings and discrete machine state
codes. No free-text, no JSON payloads. All values are numeric (float sensor channels + integer
state codes), so parsing to multivariate time series is natural and the FRS benchmark is a
reasonable scientific proxy.

Tracks:
- **Track A** — Fix the FRS pipeline (immediate; core learning problems).
- **Track B** — PLC log bridge (connect real/synthetic PLC logs to the FRS evaluation stack).
- **Track C** — LLM embedding (low priority for PLC; only relevant if alarm text or state labels
  exist in the logs).

**Do not start any experiment listed here without first checking the RETROSPECTIVE.md section
"What NOT to Retry".**

---

## Track A — Fix the FRS / Time-Series Pipeline

These experiments address the **known fundamental problems** in the Phase 3 baseline and are the
most actionable next steps. They all plug into the existing RQ1 infrastructure.

### A1: Factor-aware positive pairs

**Problem addressed**: Training–eval mismatch (#1), positive-pair misalignment (#2).

**Idea**: Replace temporal-neighbour pairs with pairs drawn from windows that share the same
mode (or the same spectral family, or the same coupling level). This can be done without
labels if the FRS dataset is used: generate pairs by sampling windows from different
trajectories that were generated under the same regime parameters.

**Design choices to test**:
- Same `mode_id` across trajectories (strictest).
- Same `spectral_id` (partially supervised, easier positives).
- Mixed: same mode within trajectory + cross-trajectory same-mode.

**Expected outcome**: Improved clustering ARI for mode and spectral factors; improved global
structure without sacrificing retrieval.

**Infrastructure**: Modify `examples/fsss/train_tcn_hybrid.py` or
`experiments/rq1_recoverability/scripts/04_train_tcn_encoder.py` to accept a
`positive_pair_strategy` config flag. Use the same evaluation pipeline (script 06).

**Evaluation baseline**: TS2Vec e120 (mode probe 0.514, spectral 0.767, coupling 0.713,
mode ARI 0.157). A new method must beat this on all three probe scores AND improve mode ARI.

---

### A2: Explicit load regression auxiliary objective

**Problem addressed**: Load not learned (#3).

**Idea**: Add a reconstruction head that predicts per-window mean load from the embedding.
This can be treated as a self-supervised auxiliary objective if mean load is derived from the
signal directly (e.g. as the mean absolute value, which correlates with FRS load).

**Design choices to test**:
- Auxiliary regression on synthetic load (use FRS ground truth as weak label during training).
- Auxiliary regression on estimated load (mean absolute signal value, no label).
- β-weighting: weight the load auxiliary loss against contrastive/reconstruction terms.

**Expected outcome**: Load R² > 0.1 (any positive result is a new finding; current best is ≤ 0).

**Caveat**: Using ground-truth load during training violates fully unsupervised framing. Test
both variants (with and without label) and report separately. The unsupervised proxy (mean abs
value) is more principled for the original goal.

---

### A3: Held-out factor combination OOD splits

**Problem addressed**: No OOD evaluation (#7).

**Idea**: Create splits where certain (spectral_id, coupling_id) combinations are held out
from training entirely. Evaluate how well embeddings generalise to unseen regime combinations.

**Design choices**:
- Hold out 2 of the 12 mode combinations from training.
- Hold out an entire coupling level (e.g. high coupling).
- Hold out a spectral family (e.g. quasi-aperiodic).

**Expected outcome**: OOD probe accuracy and retrieval lower than in-distribution—this is
expected and confirms the evaluation is meaningful. The question is whether the degradation
is catastrophic or graceful.

**Infrastructure**: Extend `experiments/rq1_recoverability/scripts/02_create_dataset_splits.py`
with an `--ood_holdout` flag and a new split config.

---

### A4: Multi-seed reporting for all new methods

**Problem addressed**: Single-seed instability (#8).

**Rule**: Any new method run under Track A must be evaluated with at least 3 seeds (e.g. 42,
53, 64) and report mean ± std. Results from a single seed cannot be compared to MOMENT's
multi-seed average.

**Note**: TS2Vec e120 was run with 1 seed. Its results should be reproduced with 3 seeds
before using it as the definitive comparison baseline.

---

### A5: Segment-level evaluation and change-point detection

**Problem addressed**: Transition smearing (#9), training–eval mismatch (#1).

**Idea**: Evaluate the embeddings on a segment-level task—given a trajectory embedding
sequence, detect where regime boundaries occur—rather than only window-level probes. This
makes the evaluation task closer to what mode-change detection actually requires.

**Methods to try**:
- Unsupervised change-point detection on the embedding sequence (e.g. BOCPD, kernel-based CPD).
- Segment clustering: assign windows to segments, evaluate whether boundaries align with
  ground-truth regime changes.

**Infrastructure**: Add a `change_point_detection.py` evaluation module under
`experiments/rq1_recoverability/src/rq1/evaluation/` or `src/version2/evaluation/`.
Add a boundary F1 metric and segment ARI to the evaluation suite.

---

### A6: Transition-aware loss term

**Problem addressed**: Transition smearing (#9).

**Idea**: Penalise the model for embedding transition windows close to stable-regime windows.
A transition-aware margin loss: push transition-window embeddings away from the nearest
stable-window embeddings of both the preceding and following regimes.

**Prerequisite**: Implement A5 first (segment-level evaluation), so the effect of this loss
can be measured.

---

## Track B — PLC Log Bridge

PLC logs are numeric-heavy (sensor floats + integer state codes), so parsing to multivariate
time series is a natural step, not a lossy approximation. This track connects real or realistic
synthetic PLC logs to the existing FRS evaluation infrastructure.

### B1: Define a synthetic PLC log generator

**Task**: Build a generator that emits FRS-like factorised regime structure in PLC log format.

**Format**: CSV rows with columns `timestamp, channel_id, value`. No text, no JSON.
Channel types:
- Float channels: sensor readings (temperature, pressure, flow rate, motor speed, …).
- Integer channels: machine state codes (0=STOPPED, 1=IDLE, 2=RUNNING, 3=FAULT, …).
- Boolean channels: alarm bits, interlock flags.

**Design**:
- Each FRS regime maps to a characteristic set of sensor distributions and a dominant state code.
- Transition windows produce mixed states and transient sensor behaviour.
- Channels sample at potentially different rates (e.g. fast sensors at 10 Hz, slow sensors at 1 Hz).
- State code changes are logged as events (sparse, not periodic).

**Purpose**: gives a directly comparable PLC-format version of the FRS dataset so the same
factor-recovery evaluation applies, and allows testing whether the parse pipeline introduces
any information loss.

---

### B2: PLC log → multivariate time series parsing

**Task**: Define and implement the canonical parse pipeline.

**Steps**:
1. **Pivot**: reshape long-format `(timestamp, channel_id, value)` to wide-format
   `(timestamp, channel_1, channel_2, …)`.
2. **Resample**: interpolate or forward-fill to a fixed time grid (configurable rate).
3. **Encode state codes**: keep integer state codes as an additional channel; optionally
   one-hot encode for models that cannot handle mixed types.
4. **Window**: extract fixed-length windows with configurable stride.
5. **Leakage check**: verify trajectory-level split metadata is preserved through the parse.

**Output**: NPZ files in the same format as `frs_clean_vnext_long` — feeds directly into
script 06 (evaluate_recoverability) without changes.

**Question to answer**: Does the round-trip (FRS generate → emit as PLC log → parse back to
time series) preserve the factor structure to within numerical noise? If the round-trip is
lossless, the FRS benchmark is a valid proxy and Track B and Track A share evaluation infrastructure.

---

### B3: Mixed discrete/continuous channel handling

**Task**: Test how methods handle a channel matrix that mixes continuous sensor readings with
discrete state codes (integers 0–N).

**Issue**: State codes are not ordinal. A model treating them as continuous values will learn
spurious distances (e.g. STOPPED=0 and RUNNING=2 seem "close", but FAULT=3 is in between).
Options:
- Pass state codes as-is (current FRS approach).
- One-hot encode state channels before windowing (increases dimensionality).
- Treat state transitions as separate event-sequence features fed into a parallel encoder.

**Evaluation**: Compare factor probe accuracy and clustering ARI across encoding strategies.
Expect state-code encoding to matter most for mode_id recovery.

---

### B4: Real PLC dataset pilot (if data available)

**Task**: Once the parse pipeline is validated on synthetic data, apply it to a real PLC dataset.

**Candidate public datasets** (no access verified — check availability):
- SWAT (Secure Water Treatment), iTrust, Singapore: industrial process, sensor + actuator logs.
- BATADAL (Battle of the Attack Detection ALgorithms): water distribution system.
- Tennessee Eastman Process (TEP): chemical plant simulation, widely used in anomaly detection.

**Evaluation**: Apply the RQ1 evaluation suite. Report which factors are recoverable in real
data vs synthetic, and note where the FRS assumptions break down.

---

## Track C — LLM-based Embedding (Low Priority for PLC)

LLM text embeddings are **not a primary direction** for PLC logs because there is no meaningful
text content. This track is retained only for the specific case where PLC logs contain textual
alarm descriptions or operator annotations alongside the numeric data.

### C1: Alarm text embedding (conditional)

**Applicable only if**: the target PLC system includes a textual alarm log alongside numeric
sensor data (e.g. alarm code + free-text description: `"High temperature fault on Motor_01"`).

**Task**: Extract window-level embeddings from alarm text using a sentence-transformer model
(`all-MiniLM-L6-v2` as the baseline). Compare factor-recovery performance against the numeric
pipeline (Track A/B).

**Expected finding**: for systems where alarms are rare and repetitive, LLM embeddings will
carry little information beyond the alarm code itself. The numeric sensor data will dominate.
Only pursue if preliminary analysis shows alarm vocabulary is diverse and mode-correlated.

---

### C2: State label annotation embedding (conditional)

**Applicable only if**: machine state codes map to meaningful label strings that a domain expert
has defined (e.g. state 2 = "High-speed cutting cycle"). In that case an LLM embedding of
state labels can be used to initialise a state-code embedding lookup, providing richer
representations than raw integers.

**Task**: Embed state label strings → use as fixed embeddings for state code channels in the
time-series model. Compare to integer encoding and one-hot encoding (B3).

**Note**: This is a feature-engineering step, not a replacement for the time-series model.

---

## Prioritisation

| Priority | Experiment | Effort | Expected Impact |
|----------|------------|--------|-----------------|
| 1 | A1: Factor-aware positive pairs | Medium | High — directly fixes core mismatch |
| 2 | A4: Multi-seed for TS2Vec | Low | High — required for credible comparison |
| 3 | B1 + B2: Synthetic PLC log generator + parse pipeline | Medium | High — closes the gap to the real domain |
| 4 | A2: Load regression auxiliary | Medium | High — only open quantitative gap |
| 5 | B3: Mixed discrete/continuous channel handling | Low | Medium — directly relevant to PLC state codes |
| 6 | A3: OOD splits | Low | Medium — reveals robustness gap |
| 7 | A5: Segment-level / CPD evaluation | Medium | Medium — new evaluation dimension |
| 8 | B4: Real PLC dataset pilot | High | High — but depends on data availability |
| 9 | A6: Transition-aware loss | Medium | Low/Medium — depends on A5 result |
| 10 | C1/C2: LLM alarm/label embedding | Low | Low — only if PLC logs contain alarm text |

---

## Reporting Standards for Next Experiments

Every new experiment must:
1. Register a run entry in `experiments/registry/` using the template.
2. Report 3+ seeds or explain why single-seed is sufficient (it usually is not).
3. Include Baseline FFT as a comparison point in every table.
4. Report probe balanced accuracy, clustering ARI, and retrieval precision@10 together.
5. Run on both `frs_clean_vnext_long` and `frs_noisy_vnext_long`.
6. Note explicitly whether the result beats the TS2Vec e120 baseline.
