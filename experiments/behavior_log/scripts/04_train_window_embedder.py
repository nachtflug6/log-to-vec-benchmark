"""Stage 04: train a window embedder."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.data.windowing import WindowDatasetBuilder
from behavior_log.models import PCAWindowEmbedder, build_sequence_model
from behavior_log.utils.io import load_yaml, save_json


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "pca_window_embedder"
    cfg = load_yaml(ROOT / "configs" / "models" / f"{config_name}.yaml")
    windows = WindowDatasetBuilder.load_bundle(cfg["window_file"])
    cfg["event_type_vocab_size"] = int(np.max(windows.X[:, :, 0])) + 1
    cfg["component_vocab_size"] = int(np.max(windows.X[:, :, 1])) + 1
    cfg["severity_vocab_size"] = int(np.max(windows.X[:, :, 2])) + 1
    cfg["state_vocab_size"] = int(np.max(windows.X[:, :, 3])) + 1

    train_mask = windows.split == "train"
    model_type = str(cfg.get("model_type", "pca"))
    if model_type == "pca":
        model = PCAWindowEmbedder(
            n_components=int(cfg["n_components"]),
            whiten=bool(cfg["whiten"]),
        )
        model.fit(windows.X[train_mask], windows.mask[train_mask])
    else:
        model = build_sequence_model(
            model_type=model_type,
            cfg=cfg,
            input_dim=int(windows.X.shape[2]),
            window_length=int(windows.X.shape[1]),
        )
        model.fit(
            windows.X[train_mask],
            windows.mask[train_mask],
            trajectory_id=windows.trajectory_id[train_mask],
            window_start=windows.window_start[train_mask],
        )

    output_dir = Path(cfg["output_dir"])
    model.save(output_dir / cfg["model_file"])
    save_json(model.training_summary(), output_dir / cfg["training_summary_file"])

    print(f"Trained {model_type} window embedder on {int(train_mask.sum())} windows")
    print(f"Saved model: {output_dir / cfg['model_file']}")


if __name__ == "__main__":
    main()
