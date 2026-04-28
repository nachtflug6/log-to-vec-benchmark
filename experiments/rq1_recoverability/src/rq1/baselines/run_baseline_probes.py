"""Compute simple baseline representations and latent-factor probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rq1.baselines.features import build_feature_set


CLASSIFICATION_TARGETS = ["mode_id", "spectral_id", "coupling_id", "is_transition_window"]
REGRESSION_TARGETS = ["mean_load"]


def load_split(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def classifier(kind: str):
    if kind == "linear":
        return Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=42))])
    if kind == "rbf":
        return Pipeline([("scaler", StandardScaler()), ("clf", SVC(kernel="rbf", gamma="scale", random_state=42))])
    raise ValueError(f"Unsupported probe model: {kind}")


def regressor(kind: str):
    if kind == "linear":
        return Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    if kind == "rbf":
        return Pipeline([("scaler", StandardScaler()), ("reg", SVR(kernel="rbf", gamma="scale"))])
    raise ValueError(f"Unsupported probe model: {kind}")


def probe_classification(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, kind: str) -> dict[str, Any]:
    if len(np.unique(train_y)) < 2 or len(np.unique(test_y)) < 2:
        return {"skipped": True, "skipped_reason": "train or test split has fewer than two classes"}
    model = classifier(kind)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    return {
        "skipped": False,
        "accuracy": float(accuracy_score(test_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
    }


def probe_regression(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, kind: str) -> dict[str, float]:
    model = regressor(kind)
    model.fit(train_x, train_y)
    pred = model.predict(test_x)
    return {
        "mae": float(mean_absolute_error(test_y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(test_y, pred))),
        "r2": float(r2_score(test_y, pred)),
    }


def run_feature_set(
    train_split: dict[str, np.ndarray],
    test_split: dict[str, np.ndarray],
    feature_set: str,
    probe_models: list[str],
) -> dict[str, Any]:
    train_features = build_feature_set(train_split["X"], feature_set)
    test_features = build_feature_set(test_split["X"], feature_set)
    targets: dict[str, Any] = {}
    for target in CLASSIFICATION_TARGETS:
        if target not in train_split or target not in test_split:
            continue
        targets[target] = {
            kind: probe_classification(train_features, train_split[target], test_features, test_split[target], kind)
            for kind in probe_models
        }
    for target in REGRESSION_TARGETS:
        if target not in train_split or target not in test_split:
            continue
        targets[target] = {
            kind: probe_regression(train_features, train_split[target], test_features, test_split[target], kind)
            for kind in probe_models
        }
    return {
        "train_feature_shape": list(train_features.shape),
        "test_feature_shape": list(test_features.shape),
        "targets": targets,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RQ1 baseline feature probes.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--feature_sets", nargs="+", default=["fft", "summary", "raw_flatten"])
    parser.add_argument("--probe_models", nargs="+", default=["linear", "rbf"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_split = load_split(args.train_file)
    test_split = load_split(args.test_file)
    output = {
        "train_file": args.train_file,
        "test_file": args.test_file,
        "num_train": int(train_split["X"].shape[0]),
        "num_test": int(test_split["X"].shape[0]),
        "window_shape": list(train_split["X"].shape[1:]),
        "results": {
            feature_set: run_feature_set(train_split, test_split, feature_set, args.probe_models)
            for feature_set in args.feature_sets
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_probe_results.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved baseline probe results to: {output_dir / 'baseline_probe_results.json'}")


if __name__ == "__main__":
    main()
