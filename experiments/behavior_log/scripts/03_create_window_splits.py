"""Stage 03: build windows and trace-level splits."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.data.windowing import WindowDatasetBuilder
from behavior_log.preprocessing.event_preprocessor import EventLogPreprocessor
from behavior_log.utils.io import load_yaml, save_json


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    cfg = load_yaml(ROOT / "configs" / "windowing" / f"{config_name}.yaml")
    bundle = EventLogPreprocessor.load_bundle(cfg["event_feature_file"])

    builder = WindowDatasetBuilder(
        window_length=int(cfg["window_length"]),
        stride=int(cfg["stride"]),
        pad_short_traces=bool(cfg["pad_short_traces"]),
        seed=int(cfg["seed"]),
    )
    windows = builder.build_windows(bundle)
    windows, split_manifest = builder.assign_trace_splits(
        windows=windows,
        train_ratio=float(cfg["train_ratio"]),
        val_ratio=float(cfg["val_ratio"]),
        test_ratio=float(cfg["test_ratio"]),
    )

    output_dir = Path(cfg["output_dir"])
    builder.save_bundle(windows, output_dir / cfg["window_file"])
    builder.save_split_manifest(split_manifest, output_dir / "split_manifest.json")

    split_counts = {
        split_name: int((windows.split == split_name).sum())
        for split_name in ("train", "val", "test")
    }
    summary = {
        "n_windows": int(len(windows.X)),
        "window_shape": list(windows.X.shape[1:]),
        "split_counts": split_counts,
    }
    save_json(summary, output_dir / "window_summary.json")

    print(f"Created {summary['n_windows']} windows")
    print(f"Window shape: {summary['window_shape']}")
    print(f"Saved: {output_dir / cfg['window_file']}")


if __name__ == "__main__":
    main()
