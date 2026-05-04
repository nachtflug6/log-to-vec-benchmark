"""Stage 04: compute a simple baseline window representation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.baselines.window_representations import WindowRepresentationBuilder
from behavior_log.data.windowing import WindowDatasetBuilder
from behavior_log.utils.io import load_yaml, save_json


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "pca"
    cfg = load_yaml(ROOT / "configs" / "representations" / f"{config_name}.yaml")
    windows = WindowDatasetBuilder.load_bundle(cfg["window_file"])

    builder = WindowRepresentationBuilder(
        method=str(cfg["method"]),
        n_components=int(cfg.get("n_components", 8)),
        whiten=bool(cfg.get("whiten", False)),
    )
    result = builder.fit_transform(windows)

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = output_dir / "window_embeddings.npz"
    np.savez_compressed(
        embedding_path,
        embeddings=result.embeddings,
        window_behavior_id=windows.window_behavior_id,
        trajectory_id=windows.trajectory_id,
        is_transition_window=windows.is_transition_window,
        split=windows.split,
        window_start=windows.window_start,
    )
    builder.save(output_dir / cfg["state_file"])
    save_json(result.summary, output_dir / cfg["summary_file"])

    print(f"Computed {cfg['method']} embeddings for {len(result.embeddings)} windows")
    print(f"Saved: {embedding_path}")


if __name__ == "__main__":
    main()
