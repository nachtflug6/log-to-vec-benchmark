# Merge Ledger: Unified Mode-Change Benchmark

Target branch: unified-mode-change-benchmark
Base branch: main
Date initialized: 2026-04-28

## Source Branches

| Branch | Head SHA | Last commit title | Status | Notes |
|---|---|---|---|---|
| experiment1 | 56c4cf5 | update | partially-integrated | code/config included, artifacts deferred |
| version2 | 59466e4 | moment method | partially-integrated | code/config included, outputs deferred |
| contrastive-phase1 | 432f325 | Update dataset.py | partially-integrated | core `log_to_vec` contrastive path included, `src/version1` deferred |

## Merge Checklist

- [x] Map file-level diffs for each source branch against main
- [x] Classify changes by area (data/model/eval/scripts/docs)
- [x] Resolve conflicts and choose canonical implementation per area
- [x] Record included commits/files with rationale
- [x] Record excluded commits/files with rationale
- [ ] Run smoke tests after each merge step

## Conflict Matrix

| Area | experiment1 | version2 | contrastive-phase1 | Canonical choice | Rationale |
|---|---|---|---|---|---|
| Data pipeline | `rq1` generators and split tooling included | FSSS loaders included | contrastive dataset+augmentations included | keep all code paths under explicit namespaces | supports parallel baselines and objective comparisons |
| Modeling objective | baseline probes + TCN recoverability included | hybrid TCN objective included | LSTM contrastive objective included | baseline + contrastive + hybrid coexist | preserves branch research diversity for benchmarking |
| Evaluation | recoverability tables and scripts included | eval_v2 and fsss_eval included | contrastive metrics included | consolidate under experiment-run logging | enables comparable reports across methods |
| Scripts and docs | RQ1 scripts/docs included | FSSS scripts included | contrastive scripts included | include executable scripts, defer generated outputs | reproducibility without repository bloat |
| Infrastructure | artifacts deferred | outputs deferred | legacy `src/version1` deferred | code-first branch policy | keeps unified branch maintainable |

## Decision Log

### 2026-04-28

- Initialized integration branch and source branch inventory.
- Produced file-level comparison against all three source branches.
- Current answer to "all files added?": no, source branch content is still not merged.
- Started selective integration pass from `contrastive-phase1` (code/config paths only).
- Pulled into working tree: `configs/contrastive_toy.yaml`, contrastive examples, and `src/log_to_vec/{data,evaluation,models,training}` contrastive modules.
- Explicitly deferred in first pass: `src/version1/**` legacy subtree pending keep-or-drop decision.
- Updated `examples/train_contrastive.py` imports and batch key usage to align with merged `log_to_vec` modules.
- Started selective integration pass from `version2` (code/config only).
- Pulled into working tree: `configs/ts2vec_ett.yaml`, `examples/fsss/**`, `examples/moment/moment_sine_generator.py`, `src/moment/data/**`, and `src/version2/**`.
- Explicitly deferred in first pass: `outputs/**`, root scratch files (`scr.txt`, `test.py`), and other generated artifacts.

## Branch Comparison Snapshot (2026-04-28)

### Commits not yet integrated

- experiment1: 4 commits not in integration branch
- version2: 4 commits not in integration branch
- contrastive-phase1: 7 commits not in integration branch

### File-level delta summary

- experiment1: 177 files differ
- version2: 181 files differ
- contrastive-phase1: 20 files differ

### Preliminary content breakdown

- experiment1: about 43 code or config files, about 123 artifact or result files
- version2: about 26 code or config files, about 151 artifact or result files
- contrastive-phase1: 20 code or config files, 0 obvious artifact files

### Notable missing code or config scopes

- experiment1: `experiments/rq1_recoverability` pipeline (configs, scripts, src helpers), plus `src/version2/evaluation/eval_v2.py`
- version2: `examples/fsss`, `src/version2/*`, `src/moment/data/*`, and `configs/ts2vec_ett.yaml`
- contrastive-phase1: `src/log_to_vec/models/contrastive.py`, `src/log_to_vec/training/contrastive_losses.py`, `src/log_to_vec/data/contrastive_dataset.py`, and related examples/configs

### Merge strategy recommendation

- Integrate code-first and config-first changes from each branch.
- Do not merge historical `outputs/` and large `artifacts/` directories into the unified research branch; preserve via external storage or selective archival index.

### Selective integration progress (working tree, pre-commit)

- `contrastive-phase1`: 10 of 20 branch-diff files remain absent, all under `src/version1/**` legacy scope.
- `version2`: 154 of 181 branch-diff files remain absent, dominated by `outputs/**` artifacts plus root scratch files.
- `experiment1`: 124 of 177 branch-diff files remain absent, dominated by `experiments/rq1_recoverability/artifacts/**` plus scratch leftovers.

Interpretation:
- Core code/config paths from all three branches are now being staged into the integration branch.
- Remaining gaps are mostly large generated artifacts and legacy folders intentionally deferred from first-pass merge.

### First-pass include or exclude decisions

Include now:
- `src/log_to_vec/*` contrastive modules and examples/configs from `contrastive-phase1`
- `src/version2/**`, `src/moment/data/**`, `examples/fsss/**`, and related configs from `version2`
- `experiments/rq1_recoverability/{configs,manifests,scripts,src,reports/result_tables}` and MOMENT export helper from `experiment1`

Exclude in this pass:
- `src/version1/**` legacy subtree from `contrastive-phase1`
- `outputs/**` generated files from `version2`
- `experiments/rq1_recoverability/artifacts/**` generated datasets and checkpoints from `experiment1`
- root scratch files: `scr.txt`, `test.py`, `baseline_features.py`
