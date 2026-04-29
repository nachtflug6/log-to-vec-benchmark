# Historical Results And Artifacts

This branch intentionally commits generated artifacts from earlier experiment branches so prior work is inspectable.

## Sources

- `origin/version2:outputs/**` -> restored to `outputs/**`.
- `origin/experiment1:experiments/rq1_recoverability/artifacts/**` -> restored to `experiments/rq1_recoverability/artifacts/**`.
- `origin/contrastive-phase1:src/version1/**` -> archived under `archive/legacy/contrastive_phase1/src/version1/**`.
- Branch scratch files -> archived under `archive/branch_scratch/**`.

## Artifact Policy

Historical artifacts are committed as evidence. Future generated outputs remain ignored by default through `.gitignore`; intentionally curated artifacts should be force-added with a clear commit message and an accompanying run metadata entry.

## Notable Historical Result Themes

Early simplified datasets showed high spectral probe and retrieval performance, but those results were partly driven by simple frequency structure. Later RQ1 work moved to factorized regimes with mode, spectral family, coupling, transitions, and load.

Observed patterns from context notes:

- MOMENT and TS2Vec-style embeddings recover spectral structure better than load.
- Coupling is recoverable but weaker than spectral structure.
- Retrieval can be stronger than clustering, suggesting local structure without clean global organization.
- Baselines are competitive and should remain part of every report.
- Noisy settings reduce factor recovery and expose robustness limits.

Use this index as a starting point, then inspect the JSON/Markdown files inside the artifact folders for exact numbers.
