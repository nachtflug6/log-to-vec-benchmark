"""Generate PCA-over-time worm plots for the active RQ1 methods.

Layout:
- rows: MOMENT / TS2Vec tuned e120 / masked multiscale reconstruction e160
- cols: spectral_id / coupling_id / is_transition_window

Each row fits PCA on the full test embedding set for that method, then draws a
time-ordered trajectory worm in the 2D PCA plane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.decomposition import PCA


METHOD_SPECS = [
    {
        "label": "MOMENT",
        "run_suffix_clean": "frs_clean_moment_vnext_long_fair",
        "run_suffix_noisy": "frs_noisy_moment_vnext_long_fair",
        "embedding_subdir": ["moment_pretrained", "embeddings"],
    },
    {
        "label": "TS2Vec tuned e120",
        "run_suffix_clean": "frs_clean_vnext_long_ts2vec_tuned_e120",
        "run_suffix_noisy": "frs_noisy_vnext_long_ts2vec_tuned_e120",
        "embedding_subdir": ["ts2vec_style", "embeddings"],
    },
    {
        "label": "Masked multiscale e160",
        "run_suffix_clean": "frs_clean_vnext_long_masked_multiscale_e160",
        "run_suffix_noisy": "frs_noisy_vnext_long_masked_multiscale_e160",
        "embedding_subdir": ["masked_multiscale_reconstruction", "embeddings"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate worm-plot comparison panels for active RQ1 methods.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="both",
        choices=["frs_clean_vnext_long", "frs_noisy_vnext_long", "both"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--trajectory_id",
        type=int,
        default=None,
        help="Optional explicit trajectory_id from the test split. If omitted, the script auto-selects one.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[1] / "reports" / "figures" / "worm_plots"),
    )
    return parser.parse_args()


def load_split(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {key: data[key] for key in data.files}


def load_embeddings(npz_path: Path) -> np.ndarray:
    with np.load(npz_path) as data:
        return data["embeddings"].astype(np.float32)


def resolve_paths(root: Path, dataset_name: str, seed: int) -> tuple[Path, list[dict[str, Any]]]:
    split_path = root / "artifacts" / "datasets" / dataset_name / "splits" / f"trajectory_seed{seed}" / "test_windows.npz"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing test split: {split_path}")

    dataset_key = "clean" if "clean" in dataset_name else "noisy"
    methods = []
    for spec in METHOD_SPECS:
        run_name = spec["run_suffix_clean"] if dataset_key == "clean" else spec["run_suffix_noisy"]
        embedding_path = root / "artifacts" / "runs" / run_name
        for part in spec["embedding_subdir"]:
            embedding_path = embedding_path / part
        embedding_path = embedding_path / "test_embeddings.npz"
        if not embedding_path.exists():
            raise FileNotFoundError(f"Missing embeddings for {spec['label']}: {embedding_path}")
        methods.append({"label": spec["label"], "embedding_path": embedding_path})
    return split_path, methods


def choose_trajectory(split: dict[str, np.ndarray]) -> int:
    trajectory_ids = np.unique(split["trajectory_id"]).astype(int).tolist()
    best = None
    best_score = None

    for trajectory_id in trajectory_ids:
        mask = split["trajectory_id"] == trajectory_id
        starts = split["window_start"][mask]
        order = np.argsort(starts)
        mode_seq = split["mode_id"][mask][order]
        spectral_seq = split["spectral_id"][mask][order]
        coupling_seq = split["coupling_id"][mask][order]
        transition_seq = split["is_transition_window"][mask][order].astype(int)

        mode_changes = int(np.sum(mode_seq[1:] != mode_seq[:-1]))
        spectral_changes = int(np.sum(spectral_seq[1:] != spectral_seq[:-1]))
        coupling_changes = int(np.sum(coupling_seq[1:] != coupling_seq[:-1]))
        transition_count = int(transition_seq.sum())
        num_windows = int(mask.sum())

        score = (
            spectral_changes,
            coupling_changes,
            mode_changes,
            transition_count,
            num_windows,
            -trajectory_id,
        )
        if best_score is None or score > best_score:
            best_score = score
            best = trajectory_id

    if best is None:
        raise RuntimeError("Could not select a trajectory for worm plotting.")
    return int(best)


def categorical_color_sequence(values: np.ndarray, cmap_name: str) -> tuple[np.ndarray, list[Line2D]]:
    unique = np.unique(values)
    cmap = plt.get_cmap(cmap_name, len(unique))
    lookup = {val: cmap(i) for i, val in enumerate(unique)}
    colors = np.array([lookup[val] for val in values])
    handles = [
        Line2D([0], [0], marker="o", linestyle="", color=lookup[val], label=str(int(val)), markersize=6)
        for val in unique
    ]
    return colors, handles


def plot_factor_worm(
    ax: plt.Axes,
    coords: np.ndarray,
    values: np.ndarray,
    factor_name: str,
) -> None:
    ax.plot(coords[:, 0], coords[:, 1], color="#b0b0b0", linewidth=1.2, alpha=0.9, zorder=1)

    if factor_name == "is_transition_window":
        clean_mask = ~values.astype(bool)
        transition_mask = values.astype(bool)
        ax.scatter(coords[clean_mask, 0], coords[clean_mask, 1], s=24, c="#4c78a8", alpha=0.85, zorder=2, label="clean")
        ax.scatter(
            coords[transition_mask, 0],
            coords[transition_mask, 1],
            s=34,
            c="#e45756",
            alpha=0.95,
            zorder=3,
            label="transition",
        )
        handles = [
            Line2D([0], [0], marker="o", linestyle="", color="#4c78a8", label="clean", markersize=6),
            Line2D([0], [0], marker="o", linestyle="", color="#e45756", label="transition", markersize=6),
        ]
    elif factor_name == "spectral_id":
        colors, handles = categorical_color_sequence(values, "tab10")
        ax.scatter(coords[:, 0], coords[:, 1], s=26, c=colors, alpha=0.9, zorder=2)
    elif factor_name == "coupling_id":
        colors, handles = categorical_color_sequence(values, "Dark2")
        ax.scatter(coords[:, 0], coords[:, 1], s=26, c=colors, alpha=0.9, zorder=2)
    else:
        raise ValueError(f"Unsupported factor for worm plot: {factor_name}")

    ax.scatter(coords[0, 0], coords[0, 1], marker="*", s=150, c="#111111", zorder=4)
    ax.scatter(coords[-1, 0], coords[-1, 1], marker="X", s=90, c="#111111", zorder=4)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(handles=handles, title=factor_name, loc="best", fontsize=7, title_fontsize=8, frameon=True)


def build_panel_for_dataset(
    root: Path,
    dataset_name: str,
    seed: int,
    output_dir: Path,
    requested_trajectory_id: int | None,
) -> Path:
    split_path, method_paths = resolve_paths(root, dataset_name, seed)
    split = load_split(split_path)
    trajectory_id = requested_trajectory_id if requested_trajectory_id is not None else choose_trajectory(split)

    mask = split["trajectory_id"] == trajectory_id
    if not mask.any():
        raise ValueError(f"trajectory_id={trajectory_id} not present in {split_path}")
    order = np.argsort(split["window_start"][mask])
    ordered_index = np.where(mask)[0][order]

    fig, axes = plt.subplots(
        nrows=len(method_paths),
        ncols=3,
        figsize=(14, 11),
        constrained_layout=True,
    )
    factor_names = ["spectral_id", "coupling_id", "is_transition_window"]

    for row, method in enumerate(method_paths):
        full_embeddings = load_embeddings(method["embedding_path"])
        pca = PCA(n_components=2, random_state=42)
        full_coords = pca.fit_transform(full_embeddings)
        coords = full_coords[ordered_index]

        for col, factor_name in enumerate(factor_names):
            ax = axes[row, col]
            plot_factor_worm(ax, coords, split[factor_name][ordered_index], factor_name)
            if row == 0:
                ax.set_title(factor_name)
            if col == 0:
                ax.text(
                    -0.30,
                    0.5,
                    method["label"],
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=11,
                    fontweight="bold",
                )

    dataset_label = "Clean long fair" if "clean" in dataset_name else "Noisy long fair"
    fig.suptitle(
        f"Worm plots in PCA space | {dataset_label} | trajectory_id={trajectory_id} | seed={seed}",
        fontsize=14,
        fontweight="bold",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_trajectory{trajectory_id}_worm_panel.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    meta = {
        "dataset_name": dataset_name,
        "seed": seed,
        "trajectory_id": trajectory_id,
        "test_split": str(split_path),
        "methods": [
            {
                "label": method["label"],
                "embedding_path": str(method["embedding_path"]),
            }
            for method in method_paths
        ],
        "output_image": str(output_path),
    }
    meta_path = output_dir / f"{dataset_name}_trajectory{trajectory_id}_worm_panel.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(args.output_dir)
    dataset_names = (
        ["frs_clean_vnext_long", "frs_noisy_vnext_long"]
        if args.dataset_name == "both"
        else [args.dataset_name]
    )

    for dataset_name in dataset_names:
        output_path = build_panel_for_dataset(
            root=root,
            dataset_name=dataset_name,
            seed=args.seed,
            output_dir=output_dir,
            requested_trajectory_id=args.trajectory_id,
        )
        print(f"Saved worm-plot panel to: {output_path}")


if __name__ == "__main__":
    main()
