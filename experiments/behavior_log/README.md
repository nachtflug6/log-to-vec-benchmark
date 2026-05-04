# Behavior Log

This directory contains the new end-to-end pipeline for the behavior-driven log experiment.

## Pipeline Scope

The active pipeline is:

1. generate raw logs
2. preprocess each log event into a feature vector
3. create fixed-length windows and trace-level splits
4. train a window embedding model
5. extract window embeddings
6. evaluate embeddings against behavior labels

The current baseline implementation is intentionally simple and fully controllable:

- raw logs are generated from explicit hard behavior families with multiple variants
- each trajectory is built from sampled 8-event behavior cycles, then lightly corrupted with inserted noise events
- event preprocessing is deterministic
- windows are created at trace level to avoid leakage
- the initial embedding model is a PCA baseline over flattened windows
- evaluation covers retrieval, linear probing, PCA/t-SNE views, and clustering

## Directory Layout

- `configs/`: YAML configs for generation, preprocessing, windowing, model training, and evaluation
- `scripts/`: runnable stage entrypoints in intended pipeline order
- `src/behavior_log/`: reusable experiment-specific code
- `artifacts/`: generated raw logs, processed features, windows, trained models, embeddings, and metrics
- `reports/`: notes, tables, and figures for thesis-facing outputs
- `manifests/`: lightweight run and dataset registry files

## Intended Run Order

```bash
python experiments/behavior_log/scripts/01_generate_raw_logs.py
python experiments/behavior_log/scripts/02_preprocess_event_logs.py
python experiments/behavior_log/scripts/03_create_window_splits.py
python experiments/behavior_log/scripts/04_compute_window_representation.py pca
python experiments/behavior_log/scripts/06_evaluate_embeddings.py
```

## Active Defaults

- dataset profile: `configs/datasets/default.yaml`
- preprocessing profile: `configs/preprocessing/default.yaml`
- windowing profile: `configs/windowing/default.yaml`
- model profile: `configs/models/pca_window_embedder.yaml`
- representation profiles: `configs/representations/*.yaml`
- evaluation profile: `configs/evaluation/default.yaml`

## Notes

- Keep all new behavior-log artifacts under this directory instead of mixing them with earlier exploratory files.
- The generator emits ground-truth `trajectory_id`, `cycle_id`, `cycle_behavior_id`, `behavior_id`, and `is_transition` so downstream recoverability analysis stays explicit.
- The `v3 hard_behavior_sequence` default dataset uses 4 hard behavior families, behavior variants, 3 components, 60 trajectories, 200 events per trajectory, and a small amount of inserted noise.
- The first baseline is PCA because it gives us a clean, trainable embedding stage before we introduce neural encoders.
