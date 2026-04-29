# Trace Comparison in Embedding Space

## Core idea

A log stream is a sequence of events over time. If we apply a deterministic embedding to fixed-size
windows of that stream, we get a sequence of vectors in R^d — a **trace** (trajectory) in embedding
space.

```
log stream → [window_1, window_2, ..., window_n] → [e_1, e_2, ..., e_n] ∈ R^d
```

Because the embedding is deterministic, the same log segment always produces the same point in
embedding space. This makes traces **comparable across runs** without fighting stochastic noise from
the embedding itself.

## What a trace encodes

Each operating mode of the system produces a characteristic region in embedding space. A stable mode
corresponds to a trace that stays within a compact region; a mode transition is visible as the trace
leaving one region and entering another.

For periodic signals (e.g. daily traffic cycles), the trace in the leading two PCA dimensions forms
a closed loop. This is a consequence of the structure of the embedding — PC1 and PC2 capture the
dominant oscillation and its quadrature complement — not an assumption imposed on the data. The
worm plot (PC1 vs PC2 colored by time) is a human-readable projection of the full trace.

- Pure periodic signal → circle or ellipse
- Non-sinusoidal periodic → closed Lissajous-like curve
- Noisy periodic → fuzzy annulus
- Mode transition → open spiral, non-closing trajectory, visible drift between loops

PCA or UMAP are visualization tools only. The actual object of interest is the full-dimensional
trace in R^d.

## Using known mode change points

If we have ground-truth labels for when the system switches operating mode (from deployments,
config changes, incident reports, etc.), we can:

1. **Segment** each trace by mode label.
2. **Compare same-mode segments across runs** — do they occupy the same region of R^d?
3. **Compare different-mode segments** — are mode A and mode B separable? By how much?
4. **Measure within-mode drift** — is mode A in run 3 geometrically consistent with mode A in run 1?
5. **Detect transitions** — the point where the trace leaves the mode-A region and enters mode-B.

This gives a **ground-truth-driven quality criterion** for embeddings: a good embedding should
produce mode-A traces that cluster together and are well-separated from mode-B traces, across
independent runs.

## Trace comparison metrics

| Metric | Sensitivity | Use case |
|---|---|---|
| Fréchet distance | Shape + order | Comparing two traces with consistent timing |
| DTW | Shape, time-warp tolerant | Comparing traces with phase drift |
| Hausdorff distance | Worst-case deviation | Detecting outlier windows |
| Per-mode centroid + spread | Coarse, interpretable | First-pass separability check |

## Connection to the benchmark

The existing FRS benchmark evaluates embeddings via probes (factor recovery), retrieval, and
clustering — all static, per-window tasks. Trace comparison adds a **temporal, trajectory-level**
evaluation axis:

- **Probe/retrieval/clustering:** does a single window encode the right factors?
- **Trace comparison:** does the sequence of windows produce geometrically coherent, mode-consistent
  trajectories over time?

These are complementary. An embedding could score well on probes but produce chaotic traces (poor
global temporal structure), or produce clean loops but fail to encode factor values pointwise.

## Next steps

- Define a reference trace per mode from a clean reference run.
- For each new run, compute the trace and measure Fréchet / DTW distance to the reference.
- Use the mode-change-point timestamps as segmentation boundaries.
- Visualize with PCA worm plots as a sanity check.
- Eventually: use trace distance as an embedding quality metric in the benchmark suite.
