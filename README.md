# Log-to-Vec Benchmark

AI modeling experiments for multivariate time-series and log embeddings under operating mode changes.

This repository is a research workbench for learning, testing, and tracking window-level embeddings. The central question is not just "can we classify a known mode?" but whether a learned representation captures the structure of a changing system: operating regime, spectral behavior, cross-channel coupling, transition windows, continuous load, robustness, and useful neighborhood geometry.

Start with:

- [Research questions](docs/research_questions.md)
- [Benchmark design](docs/benchmark_design.md)
- [Modeling and evaluation protocol](docs/modeling_and_evaluation.md)
- [Historical results and artifacts](docs/results_artifact_index.md)

## What This Repo Is For

The project studies self-supervised and unsupervised embeddings for windows of multivariate sequences:

```text
trajectory -> windows -> encoder -> embeddings -> probes / retrieval / clustering / reports
```

The intended first benchmark is a factorized switched state-space setting with controlled latent factors. Synthetic data is used as a scientific instrument: it provides known hidden structure, controlled noise and device shifts, trajectory-level splits, and explicit transition windows.

The repo supports:

- controlled synthetic dataset generation,
- trajectory-level splitting and leakage checks,
- baseline feature extraction,
- self-supervised encoders and pretrained embedding extraction,
- factor recoverability evaluation,
- retrieval, clustering, transition, and robustness analysis,
- experiment metadata and historical artifact tracking.

## Repository Layout

```text
configs/                         Shared model and training configs
docs/                            Research notes, protocols, and artifact index
examples/                        Runnable demos and exploratory entrypoints
experiments/registry/            Run metadata schema and templates
experiments/rq1_recoverability/  Formal RQ1 benchmark pipeline
src/log_to_vec/                  Canonical reusable package
src/version2/                    Experimental FSSS / TCN hybrid namespace
src/moment/                      MOMENT synthetic-data helpers
archive/                         Archived branch-only legacy code and scratch files
outputs/                         Historical generated outputs from prior branches
data/                            Local input data; ignored except .gitkeep
```

`log_to_vec` is the canonical reusable package. `version2`, `moment`, and `experiments/rq1_recoverability/src/rq1` are research namespaces retained because they encode useful experiment history and active benchmark components.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

The full experiment stack uses PyTorch and scikit-learn. Some optional pretrained-model paths, such as MOMENT extraction, require their own model dependencies and credentials/cache setup.

## Main Workflows

### 1. Generate And Split The RQ1 Benchmark

RQ1 asks whether unsupervised embeddings can recover latent factors in a controlled synthetic setting.

```bash
python experiments/rq1_recoverability/scripts/01_generate_frs_dataset.py \
  --dataset_name frs_clean_vnext_long

python experiments/rq1_recoverability/scripts/02_create_dataset_splits.py \
  --dataset_name frs_clean_vnext_long
```

The formal dataset family is FRS: Factorized Regime Sequence. It models discrete mode as a combination of spectral family and coupling level, plus transition windows and continuous load.

### 2. Compute Baseline Representations

```bash
python experiments/rq1_recoverability/scripts/03_compute_baseline_representations.py \
  --dataset_name frs_clean_vnext_long
```

Baselines include raw summaries, PCA-style features, FFT/statistical features in exploratory paths, and probe-ready representations. These are essential: learned embeddings are only interesting if they beat strong simple comparators for the right reasons.

### 3. Train Or Extract Embeddings

Canonical examples:

```bash
python examples/train_contrastive_toy.py --config configs/contrastive_toy.yaml
python examples/fsss/train_tcn_hybrid.py --help
python examples/moment/extract_moment_embeddings.py --help
```

Research directions currently represented in the repo include:

- TCN / TS2Vec-style contrastive encoders,
- masked or reconstruction-style objectives,
- hybrid contrastive plus reconstruction losses,
- MOMENT pretrained embeddings,
- classical feature baselines.

### 4. Evaluate Recoverability

```bash
python experiments/rq1_recoverability/scripts/06_evaluate_recoverability.py --help
python examples/fsss/run_eval_v2.py --help
```

Evaluation is multi-view:

- probes for mode, spectral factor, coupling, transition, and load,
- retrieval metrics for local structure,
- clustering metrics for global structure,
- separate reporting for clean versus transition windows,
- robustness and multi-seed stability checks.

## Experiment Tracking

Use `experiments/registry/schema.json` and `experiments/registry/templates/run_metadata.template.json` for reproducible run metadata. Generated outputs can live under `outputs/` or experiment-specific artifact folders, but new generated files are ignored by default unless they are intentionally added as curated evidence.

This branch intentionally commits historical artifacts from earlier branches so prior results are inspectable. See [Historical results and artifacts](docs/results_artifact_index.md).

## Research Positioning

The project has drifted from simple toy log demos toward a broader AI modeling benchmark. The current framing is:

- The embedding is the object being learned.
- Labels are used for evaluation and diagnosis, not self-supervised pretraining.
- Positive-pair design must match the evaluation goal.
- Trajectory-level splits are mandatory to avoid leakage.
- High retrieval alone can be misleading if it only reflects temporal-neighbor bias.
- Linear probes, clustering, retrieval, transition analysis, and robustness must be read together.

## Development Checks

```bash
python3 -m compileall -q src examples experiments
python3 -m pytest tests
```

If the local environment lacks optional ML dependencies, run lightweight checks first:

```bash
python3 examples/generate_sine_logs.py --help
python3 examples/train_contrastive.py --help
python3 examples/fsss/build_fsss_splits.py --help
```

## Contributing Guidance

- Add reusable code under `src/log_to_vec` when it is meant to be canonical.
- Keep exploratory but useful research code under an explicit experiment namespace.
- Store run metadata using the registry template.
- Avoid random window-level train/test splits for overlapping trajectory windows.
- Add or update docs when changing the scientific interpretation of an experiment.
