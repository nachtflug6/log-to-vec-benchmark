# Distilled Research Findings

Source inputs:
- context/output_1.txt.txt
- context/output_2.txt.txt

Both source files are duplicates. This document consolidates unique findings.

## Core Problem Summary

The current setup over-optimizes local temporal similarity but underperforms on global mode structure, which is required for mode-change detection and clustering.

## Key Findings

1. Objective and evaluation mismatch
Training emphasizes temporal-neighbor similarity, while evaluation expects factor-level recovery and global mode structure.

2. Positive-pair misalignment
Positive pairs are temporal neighbors instead of same-mode samples across trajectories.

3. Local structure dominates global structure
Current losses improve nearest-neighbor retrieval but do not enforce globally separable mode clusters.

4. Missing factor disentanglement
No objective separates latent factors such as spectral dynamics, coupling, and load.

5. Weak handling of continuous factors
Continuous variation (for example load) is under-learned without explicit objective pressure.

6. Transition smearing
Temporal smoothing blurs transition windows rather than modeling transitions as first-class events.

7. No real OOD stress testing
Current splits do not isolate unseen combinations strongly enough to validate robustness.

8. Shortcut risk
Strong spectral cues may dominate learning and mask deeper system structure.

9. Stability gap
Single-seed style experiments make conclusions fragile.

## Implications For Unified Roadmap

- Keep reconstruction baseline, but add explicit mode-change objectives and metrics.
- Add change-point and segment-level evaluation artifacts, not only retrieval metrics.
- Require multi-seed reporting and OOD-oriented splits in benchmark protocol.
- Include transition-sensitive analysis in mode-change pipeline.

## Actionable Next Steps

- Implement baseline unsupervised change-point detector and segment clustering module.
- Add boundary metrics and segment cluster quality metrics.
- Log each run in structured registry with seed and provenance.
- Add Alvis submit/status/collect scripts to standardize experiment execution.
