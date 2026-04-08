from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from baseline_features import build_feature_set


def load_windows(npz_path: str):
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def plot_random_samples(
    X: np.ndarray,
    mode_id: np.ndarray,
    control_value: np.ndarray | None,
    output_path: Path,
    num_samples: int = 12,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.choice(n, size=min(num_samples, n), replace=False)

    fig, axes = plt.subplots(len(idx), 1, figsize=(10, 2.2 * len(idx)), sharex=True)
    if len(idx) == 1:
        axes = [axes]

    for ax, i in zip(axes, idx):
        ax.plot(X[i, :, 0], linewidth=1.2)
        title = f"sample={i} | mode_id={int(mode_id[i])}"
        if control_value is not None:
            title += f" | control={float(control_value[i]):.3f}"
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_samples_per_mode(
    X: np.ndarray,
    mode_id: np.ndarray,
    control_value: np.ndarray | None,
    output_path: Path,
    samples_per_mode: int = 3,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    unique_modes = np.unique(mode_id)

    fig, axes = plt.subplots(len(unique_modes), 1, figsize=(10, 2.4 * len(unique_modes)), sharex=True)
    if len(unique_modes) == 1:
        axes = [axes]

    for ax, m in zip(axes, unique_modes):
        idx = np.where(mode_id == m)[0]
        chosen = rng.choice(idx, size=min(samples_per_mode, len(idx)), replace=False)

        for i in chosen:
            label = None
            if control_value is not None:
                label = f"{float(control_value[i]):.2f}"
            ax.plot(X[i, :, 0], linewidth=1.0, alpha=0.85, label=label)

        ax.set_title(f"mode_id={int(m)}", fontsize=10)
        ax.grid(alpha=0.3)
        if control_value is not None:
            ax.legend(title="control", fontsize=8)

    axes[-1].set_xlabel("time")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_control_histogram(
    control_value: np.ndarray,
    mode_id: np.ndarray,
    output_path: Path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(control_value, bins=30)
    axes[0].set_title("Control value distribution")
    axes[0].set_xlabel("control_value")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.3)

    unique, counts = np.unique(mode_id, return_counts=True)
    axes[1].bar(unique.astype(str), counts)
    axes[1].set_title("Mode/bin counts")
    axes[1].set_xlabel("mode_id")
    axes[1].set_ylabel("count")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_pca_features(
    X: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    feature_type: str = "raw_flatten",
):
    feats = build_feature_set(X, feature_type=feature_type)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(feats)

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(coords[:, 0], coords[:, 1], c=labels, s=12, alpha=0.85)
    plt.title(f"PCA on generated data features ({feature_type})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "total_explained_variance_ratio": float(np.sum(pca.explained_variance_ratio_)),
    }


def compute_basic_data_summary(X: np.ndarray, mode_id: np.ndarray, control_value: np.ndarray | None):
    summary = {
        "num_samples": int(X.shape[0]),
        "seq_len": int(X.shape[1]),
        "num_channels": int(X.shape[2]),
        "global_mean": float(X.mean()),
        "global_std": float(X.std()),
        "global_min": float(X.min()),
        "global_max": float(X.max()),
        "num_modes": int(len(np.unique(mode_id))),
        "mode_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(mode_id, return_counts=True))},
    }

    if control_value is not None:
        summary["control_min"] = float(control_value.min())
        summary["control_max"] = float(control_value.max())
        summary["control_mean"] = float(control_value.mean())
        summary["control_std"] = float(control_value.std())

    return summary


def main():
    parser = argparse.ArgumentParser(description="Visualize generated synthetic data before training.")
    parser.add_argument("--windows_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_windows(args.windows_file)
    X = data["X"].astype(np.float32)
    mode_id = data["mode_id"].astype(np.int64)

    control_value = None
    if "control_value" in data:
        control_value = data["control_value"].astype(np.float32)

    plot_random_samples(
        X=X,
        mode_id=mode_id,
        control_value=control_value,
        output_path=output_dir / "random_samples.png",
    )

    plot_samples_per_mode(
        X=X,
        mode_id=mode_id,
        control_value=control_value,
        output_path=output_dir / "samples_per_mode.png",
    )

    if control_value is not None:
        plot_control_histogram(
            control_value=control_value,
            mode_id=mode_id,
            output_path=output_dir / "control_distribution.png",
        )

    pca_raw_info = plot_pca_features(
        X=X,
        labels=mode_id,
        output_path=output_dir / "pca_raw_flatten.png",
        feature_type="raw_flatten",
    )

    pca_fft_info = plot_pca_features(
        X=X,
        labels=mode_id,
        output_path=output_dir / "pca_fft.png",
        feature_type="fft",
    )

    summary = compute_basic_data_summary(X, mode_id, control_value)
    summary["pca_raw_flatten"] = pca_raw_info
    summary["pca_fft"] = pca_fft_info

    save_json(output_dir / "data_summary.json", summary)

    print(f"Saved visualizations to: {output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()