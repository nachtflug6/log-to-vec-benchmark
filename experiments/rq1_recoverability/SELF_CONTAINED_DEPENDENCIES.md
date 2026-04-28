# RQ1 Self-Contained Dependency Map

This folder is intended to run independently from the older exploratory `examples/` and `src/version2/` code.

## Local Package Layout

- `src/rq1/generation/`: formal synthetic dataset generators and dataset registry.
- `src/rq1/data/`: `.npz` split loading, trajectory/device split logic, and PyTorch window datasets.
- `src/rq1/baselines/`: simple feature baselines and probe evaluation.
- `src/rq1/evaluation/`: unified recoverability evaluation suite.
- `src/rq1/models/`: embedding exporters and optional model-specific training scripts.
- `src/rq1/utils/`: shared utility functions.

## Files Already Local

- `src/rq1/data/fsss_data.py`: leakage-aware split creation.
- `src/rq1/data/fsss_dataset.py`: `.npz` window dataset for MOMENT-style exporters.
- `src/rq1/evaluation/eval_v2.py`: local evaluation implementation.
- `src/rq1/baselines/features.py`: local raw/summary/FFT baseline features.
- `src/rq1/baselines/run_baseline_probes.py`: local baseline probe runner.
- `src/rq1/models/extract_moment_embeddings.py`: local MOMENT embedding exporter.
- `src/rq1/generation/factorized_regime_sequence_generator.py`: FRS generator.
- `src/rq1/generation/dataset_registry.py`: dataset registry.

## Files To Restore Only If You Need The Original Legacy Behavior

If you have original versions and want exact continuity with older experiments, place them here:

- Original `eval_v2.py` -> `src/rq1/evaluation/eval_v2.py`
- Original `fsss_dataset.py` -> `src/rq1/data/fsss_dataset.py`
- Original `fsss_data.py` -> `src/rq1/data/fsss_data.py`
- Original `run_baseline_probes.py` -> `src/rq1/baselines/run_baseline_probes.py`

These local replacements are currently functional, but they may not be byte-for-byte identical to earlier `version2` code.

## Optional Legacy TCN Files

Only needed if you still want to run `04_train_tcn_encoder.py` and `05_export_tcn_embeddings.py`.

- `train_tcn_hybrid.py` -> `src/rq1/models/train_tcn_hybrid.py`
- `extract_tcn_embeddings.py` -> `src/rq1/models/extract_tcn_embeddings.py`
- Any model/loss dependencies used by those scripts should also live under `src/rq1/models/` or another local `src/rq1/` subpackage.

For current RQ1 work, the TS2Vec-style script is already self-contained in `scripts/11_run_ts2vec_recoverability.py`, so these TCN files are optional.

## Runtime Dependency Status

The core RQ1 pipeline now uses local imports:

- `01_generate_*` -> `rq1.generation.dataset_registry`
- `02_create_dataset_splits.py` -> `rq1.data.fsss_data`
- `03_compute_baseline_representations.py` -> `src/rq1/baselines/run_baseline_probes.py`
- `05_export_moment_pretrained_embeddings.py` -> `src/rq1/models/extract_moment_embeddings.py`
- `06_evaluate_recoverability.py` -> `rq1.evaluation.eval_v2`

No active RQ1 script depends on the older exploratory `examples/` or `src/version2/` paths.
