# Experiment Log

This file tracks all structured runs for reproducibility.

## Run Template

- Run ID: EXP-YYYYMMDD-XXX
- Date:
- Branch:
- Commit SHA:
- Author:
- Objective:
- Dataset:
- Label regime: unlabeled training, synthetic labels for evaluation, partial labels for evaluation
- Configuration file:
- Random seed(s):
- Runtime environment: local or alvis1
- Container image:
- Metrics summary:
- Artifacts path:
- Outcome: success or failure
- Notes:

---

## Runs

### EXP-20260428-001

- Date: 2026-04-28
- Branch: unified-mode-change-benchmark
- Commit SHA: pending
- Author: pending
- Objective: repository unification kickoff and reproducibility scaffolding
- Dataset: pending
- Label regime: unlabeled training with synthetic evaluation labels
- Configuration file: configs/toy_example.yaml
- Random seed(s): 42
- Runtime environment: local
- Container image: pending
- Metrics summary: pending
- Artifacts path: pending
- Outcome: in-progress
- Notes: initialized merge ledger, registry schema, mode-change baseline scaffold, and Alvis scripts.

---

## Historical Phase Records

These are retrospective entries for experiment phases that pre-date the structured registry.
They are less precise than the template above but capture essential provenance.

### HIST-PHASE1 — Contrastive LSTM on Toy Logs

- Date range: prior to branch `contrastive-phase1` (exact dates not recorded)
- Branch: `contrastive-phase1` (archived)
- Objective: explore LSTM contrastive learning on toy log CSV data
- Dataset: `data/toy_logs.csv` (synthetic); sine-wave logs from `examples/generate_sine_logs.py`
- Label regime: unlabeled training; ad-hoc evaluation only (no factor labels)
- Configuration file: `configs/contrastive_toy.yaml`
- Random seed(s): not recorded
- Runtime environment: local
- Metrics summary: no formal probe/retrieval/clustering metrics. Embedding statistics only:
  train 1932 samples, dim 128, mean L2 norm 6.90, not collapsed.
  See `outputs/version1/evaluation/embedding_evaluation_summary.json`.
- Artifacts path: `archive/legacy/contrastive_phase1/`, `outputs/version1/`
- Outcome: discontinued (no ground truth, temporal-neighbour pairs, no evaluation framework)
- Notes: established log preprocessor and NT-Xent loss code, now in `src/log_to_vec/`.
  Detailed analysis in `experiments/history/phase1_contrastive_lstm/report.md`.

---

### HIST-PHASE2-RUN001 — TCN Hybrid Baseline on FSSS

- Date range: branch `version2` (exact date not recorded)
- Branch: `version2` (integrated)
- Objective: train TCN hybrid (contrastive + reconstruction) on factorised synthetic data;
  compare to hand-crafted baselines
- Dataset: FSSS (simpler, earlier FRS variant); clean and noisy profiles
- Label regime: unlabeled training; factor labels used for evaluation only
- Configuration file: TCN encoder config (see `experiments/rq1_recoverability/configs/models/tcn_encoder.yaml`)
- Random seed(s): not recorded (single seed)
- Runtime environment: local (Windows, based on artifact paths using backslash)
- Metrics summary (test set, balanced accuracy):
  - mode_id probe: 0.266 (RBF) — baseline wins at 0.277
  - spectral_id probe: 0.653 (linear) — baseline wins at 0.701
  - coupling_id probe: 0.423 (linear) — learned wins marginally
  - transition probe: 0.534 (RBF) — baseline wins at 0.558
  - mean_load R²: 0.004 (near zero)
  - Retrieval R@5: 0.961 (mode), 0.983 (spectral), 0.993 (coupling) — misleadingly high
  - Clustering ARI: 0.134 (mode), 0.030 (spectral), 0.020 (coupling)
- Artifacts path: `outputs/version2/reports/run_003/`, `outputs/version2/reports/compare/`
- Outcome: negative — learned embedding does not beat baselines; training–eval mismatch confirmed
- Notes: detailed analysis in `experiments/history/phase2_tcn_fsss/report.md`.
  The eval_v2 run (outputs/eval_v2/) produced degenerate results (R²=1.0, single-class train
  splits); do not cite those numbers.

---

### HIST-PHASE3-RQ1 — Formal RQ1 Benchmark (MOMENT + TS2Vec + Baselines)

- Date range: branch `experiment1` (integrated into main 2026-04-28)
- Branch: `experiment1` (integrated)
- Objective: establish rigorous FRS-based benchmark; compare MOMENT, TS2Vec, and baselines
  on factor recoverability (RQ1)
- Dataset: `frs_clean_vnext_long`, `frs_noisy_vnext_long` (trajectory-level splits, 70/15/15)
- Label regime: unlabeled training; FRS ground-truth factors used for evaluation only
- Configuration files:
  - `experiments/rq1_recoverability/configs/datasets/frs_clean_vnext_long.yaml`
  - `experiments/rq1_recoverability/configs/models/tcn_encoder.yaml`
  - `experiments/rq1_recoverability/configs/evaluation/rq1_core.yaml`
- Random seed(s): MOMENT evaluated at seeds 42, 53, 64 (3 seeds); TS2Vec at seed 42 (1 seed)
- Runtime environment: local + Alvis HPC (smoke-tested 2026-04-29)
- Metrics summary (frs_clean_vnext_long, RBF probe balanced accuracy, best per method):
  - MOMENT:      mode 0.466±0.012, spectral 0.756±0.018, coupling 0.633±0.042, load R² −0.037
  - TS2Vec e120: mode 0.514,       spectral 0.767,       coupling 0.713,       load R² −0.006
  - Baseline FFT: mode 0.439,       spectral 0.696,       coupling 0.673
  - Avg factor (mode/spectral/coupling): FFT 0.603, MOMENT 0.618, TS2Vec e120 0.665
  - Mode ARI: MOMENT 0.151, TS2Vec e80 0.153, TS2Vec e120 0.157
  - Coupling ARI: near zero for all trained methods (0.004–0.058)
  - Load R²: ≤ 0 for all methods — not learned
- Artifacts path: `experiments/rq1_recoverability/artifacts/`, `experiments/rq1_recoverability/reports/result_tables/`
- Outcome: partial success — baseline established, fundamental problems identified
- Notes: full analysis in `experiments/history/phase3_rq1_benchmark/report.md`.
  This phase produced the current state-of-the-art numbers and the confirmed problem list.
  Do not re-run MOMENT or TS2Vec without changes; the results are already in the tables.

---

### HIST-PHASE3-SMOKE — Alvis Apptainer Smoke Deploys

- Date: 2026-04-29
- Branch: main
- Commit SHA: 639ec50
- Objective: verify Alvis HPC infrastructure (Slurm + Apptainer) can train and export a
  small contrastive model end-to-end
- Dataset: small synthetic (sequence shape 64×12×4 windows)
- Configuration: smoke deploy config (`scripts/smoke_deploy_model.py`)
- Random seed(s): not fixed
- Runtime environment: Alvis HPC (T4 GPU, PyTorch Apptainer image)
- Metrics summary: infrastructure verification only; no factor-recovery metrics
- Artifacts path: `outputs/alvis_smoke/ALVIS-SMOKE-*/`
- Outcome: success — model trained, `best_model.pt` and `test_embeddings.npy` produced
- Notes: 5 smoke runs (ALVIS-SMOKE-001 through ALVIS-SMOKE-005) visible in artifacts.
  HPC pipeline ready for production RQ1 experiments.
