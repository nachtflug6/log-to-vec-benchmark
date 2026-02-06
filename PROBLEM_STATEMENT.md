# Why Log Embeddings Matter

Industrial and software systems generate massive volumes of time-stamped logs. Turning these sequences into **embeddings** (compact vectors) makes downstream tasks practical: anomaly detection, clustering, similarity search, and predictive maintenance. The goal is to learn **regularized representations** that preserve important structure while remaining compact, stable, and generalizable.

## What We’re Studying

We want to understand **which models perform best under which constraints**. That means comparing architectures (autoencoders, Transformers, contrastive methods) while varying practical limits that matter in real deployments.

## Edge-AI Constraints (Examples)

When log vectorization is deployed near the data source (edge), constraints are often tighter than in cloud environments. Relevant constraints include:

- **Latency**: embeddings must be computed quickly for near‑real‑time alerts.
- **Memory footprint**: edge devices may only allow small models.
- **Compute budget**: limited CPU/GPU resources or no GPU at all.
- **Energy consumption**: power usage may be constrained (battery/thermal limits).
- **Model size / update cost**: frequent retraining or updates may be expensive.
- **Streaming input**: logs arrive continuously, requiring online or incremental processing.
- **Robustness**: tolerate missing values, noisy events, or drift in log patterns.

## What “Regularized Embeddings” Means Here

We want embeddings that are:

- **Compact**: small dimensional vectors that capture essential behavior.
- **Stable**: similar sequences map to similar embeddings.
- **Generalizable**: work across different scenarios and log distributions.
- **Useful**: improve downstream tasks (clustering, retrieval, anomaly detection).

This project benchmarks how different modeling choices trade off these goals.
