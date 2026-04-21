# RQ1 Recoverability

This directory contains the cleaned and reproducible pipeline for RQ1:

"Can unsupervised embeddings recover latent factors in controlled synthetic settings?"

## Scope

RQ1 focuses on controlled synthetic datasets and compares simple baselines against learned embeddings under one unified evaluation protocol.

Primary datasets:
- `moment_freq`: simplest sanity-check dataset
- `frs_clean`: formal clean benchmark
- `frs_noisy`: formal harder benchmark
- `frs_clean_v3`: balanced clean benchmark with more independent latent-factor footprints
- `frs_noisy_v3`: balanced harder benchmark with the same v3 factor design
- `frs_clean_vnext`: mechanistic clean benchmark built from latent process, dynamics, and observation layers
- `frs_noisy_vnext`: mechanistic harder benchmark with the same layered design
- `frs_clean_vnext_long`: long-window variant of the mechanistic clean benchmark
- `frs_noisy_vnext_long`: long-window variant of the mechanistic noisy benchmark

Dataset family naming:
- `FRS` = `Factorized Regime Sequence`
- The formal FRS generator now lives under `src/rq1/factorized_regime_sequence_generator.py`
- Compatibility aliases (`mode_id`, `spectral_id`, `coupling_id`) are still emitted so the shared evaluation code can be reused

Primary representations:
- raw baseline features
- PCA baseline features
- pretrained MOMENT embeddings
- optional learned embeddings for exploratory comparisons only

Primary evaluation targets:
- `mode_id`
- `spectral_id`
- `coupling_id`
- `is_transition_window`
- `mean_load`

Primary evaluation views:
- linear probe
- retrieval
- transition-specific analysis

## Directory Layout

- `configs/`: dataset, split, model, baseline, and evaluation configs
- `scripts/`: runnable pipeline entrypoints
- `src/rq1/`: reusable RQ1-specific helpers
- `artifacts/`: generated datasets, runs, embeddings, and metrics
- `reports/`: tables, figures, and experiment notes for thesis writing
- `manifests/`: lightweight registry files for datasets and runs

## Intended Run Order

1. Generate dataset artifact
2. Build leakage-aware splits
3. Compute baseline representations
4. Export pretrained MOMENT embeddings
5. Run unified recoverability evaluation
6. Run the end-to-end dataset wrapper
7. Build thesis-facing report assets

## Notes

- Keep all RQ1 experiments under this directory to avoid mixing with earlier exploratory outputs.
- Prefer trajectory-level splits for synthetic sequence windows.
- Only compare methods when they use the same split and the same valid targets.
- Start with `moment_freq` as a sanity check before moving to the formal `FRS` datasets.
- Use the workspace virtual environment Python at `venv\Scripts\python.exe` for all end-to-end runs in this directory.
