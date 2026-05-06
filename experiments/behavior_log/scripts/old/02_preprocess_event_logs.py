"""Stage 02: preprocess raw logs into event-level feature vectors."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.preprocessing.event_preprocessor import EventLogPreprocessor
from behavior_log.utils.io import load_yaml, save_json


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "default"
    cfg = load_yaml(ROOT / "configs" / "preprocessing" / f"{config_name}.yaml")
    raw_df = pd.read_csv(cfg["raw_log_csv"])
    preprocessor = EventLogPreprocessor()
    bundle = preprocessor.fit_transform(raw_df)

    output_dir = Path(cfg["output_dir"])
    EventLogPreprocessor.save_bundle(bundle, output_dir / cfg["event_feature_file"])
    preprocessor.save_state(output_dir / cfg["preprocessor_state_file"])

    summary = {
        "n_events": int(bundle.X.shape[0]),
        "feature_dim": int(bundle.X.shape[1]),
        "n_trajectories": int(len(set(bundle.trajectory_id.tolist()))),
        "feature_names": bundle.feature_names.tolist(),
    }
    save_json(summary, output_dir / "preprocessing_summary.json")

    print(f"Preprocessed {summary['n_events']} events")
    print(f"Feature dim: {summary['feature_dim']}")
    print(f"Saved: {output_dir / cfg['event_feature_file']}")


if __name__ == "__main__":
    main()
