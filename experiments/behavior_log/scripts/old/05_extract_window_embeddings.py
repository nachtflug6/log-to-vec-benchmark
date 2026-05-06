"""Stage 05: extract embeddings from a trained window model."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.data.windowing import WindowDatasetBuilder
from behavior_log.models import PCAWindowEmbedder, load_sequence_model
from behavior_log.utils.io import load_yaml, save_json


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "pca_window_embedder"
    model_cfg = load_yaml(ROOT / "configs" / "models" / f"{config_name}.yaml")
    windows = WindowDatasetBuilder.load_bundle(model_cfg["window_file"])
    model_type = str(model_cfg.get("model_type", "pca"))
    if model_type == "pca":
        model = PCAWindowEmbedder.load(Path(model_cfg["output_dir"]) / model_cfg["model_file"])
        embeddings = model.transform(windows.X, windows.mask)
    else:
        model = load_sequence_model(
            Path(model_cfg["output_dir"]) / model_cfg["model_file"],
            model_type=model_type,
            device=str(model_cfg.get("device", "cpu")),
        )
        embeddings = model.transform(windows.X, windows.mask)
    output_dir = Path(model_cfg["output_dir"])
    embedding_path = output_dir / "window_embeddings.npz"

    np.savez_compressed(
        embedding_path,
        embeddings=embeddings,
        window_behavior_id=windows.window_behavior_id,
        trajectory_id=windows.trajectory_id,
        is_transition_window=windows.is_transition_window,
        split=windows.split,
        window_start=windows.window_start,
    )
    save_json(
        {
            "n_windows": int(len(embeddings)),
            "embedding_dim": int(embeddings.shape[1]),
            "embedding_file": str(embedding_path),
        },
        output_dir / "embedding_summary.json",
    )

    print(f"Extracted embeddings for {len(embeddings)} windows")
    print(f"Saved: {embedding_path}")


if __name__ == "__main__":
    main()
