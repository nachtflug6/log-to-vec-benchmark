from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression, Ridge

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from version2.evaluation.baseline_features import build_feature_set


def load_split(npz_path: str) -> Dict[str, np.ndarray]:
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def prepare_labels(split: Dict[str, np.ndarray], target: str) -> np.ndarray:
    if target not in split:
        raise ValueError(f"Target '{target}' not found in split keys: {list(split.keys())}")
    return split[target]


def fit_classifier(kind: str):
    if kind == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=5000, solver="lbfgs", random_state=42)),
        ])
    if kind == "rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(C=1.0, kernel="rbf", gamma="scale", random_state=42)),
        ])
    raise ValueError(f"Unknown classifier kind: {kind}")


def fit_regressor(kind: str):
    if kind == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])
    if kind == "rbf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(C=1.0, kernel="rbf", gamma="scale")),
        ])
    raise ValueError(f"Unknown regressor kind: {kind}")


def run_classification(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    kind: str,
) -> Dict[str, float]:
    train_classes = np.unique(train_y)
    test_classes = np.unique(test_y)

    if len(train_classes) < 2:
        return {
            "skipped": True,
            "skipped_reason": f"train split has only one class: {train_classes.tolist()}",
        }

    if len(test_classes) < 2:
        return {
            "skipped": True,
            "skipped_reason": f"test split has only one class: {test_classes.tolist()}",
        }

    model = fit_classifier(kind)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    return {
        "skipped": False,
        "accuracy": float(accuracy_score(test_y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(test_y, pred)),
    }


def run_regression(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    kind: str,
) -> Dict[str, float]:
    model = fit_regressor(kind)
    model.fit(train_X, train_y)
    pred = model.predict(test_X)
    return {
        "mae": float(mean_absolute_error(test_y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(test_y, pred))),
        "r2": float(r2_score(test_y, pred)),
    }


def build_target_map(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray]) -> Dict[str, Tuple[str, str]]:
    common = set(train.keys()).intersection(set(test.keys()))
    mapping = {}
    for key in ["mode_id", "spectral_id", "coupling_id", "mean_load", "is_transition_window"]:
        if key in common:
            mapping[key] = ("regression" if key == "mean_load" else "classification", key)
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple baseline probes on FSSS splits.")
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--feature_sets",
        type=str,
        nargs="+",
        default=["fft", "summary", "raw_flatten"],
        choices=["fft", "summary", "raw_flatten"],
    )
    parser.add_argument(
        "--probe_models",
        type=str,
        nargs="+",
        default=["linear", "rbf"],
        choices=["linear", "rbf"],
    )
    parser.add_argument(
        "--targets",
        type=str,
        nargs="+",
        default=["mode_id", "spectral_id", "coupling_id", "mean_load", "is_transition_window"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split(args.train_file)
    test = load_split(args.test_file)

    target_map = build_target_map(train, test)
    selected_targets = [t for t in args.targets if t in target_map]

    train_X_raw = train["X"].astype(np.float32)
    test_X_raw = test["X"].astype(np.float32)

    summary = {
        "train_file": args.train_file,
        "test_file": args.test_file,
        "num_train": int(len(train_X_raw)),
        "num_test": int(len(test_X_raw)),
        "window_shape": list(train_X_raw.shape[1:]),
        "results": {},
    }

    for feature_type in args.feature_sets:
        print(f"\n=== Feature set: {feature_type} ===")
        train_feat = build_feature_set(train_X_raw, feature_type)
        test_feat = build_feature_set(test_X_raw, feature_type)

        feature_results = {
            "train_feature_shape": list(train_feat.shape),
            "test_feature_shape": list(test_feat.shape),
            "targets": {},
        }

        for target in selected_targets:
            task_type, key = target_map[target]
            train_y = prepare_labels(train, key)
            test_y = prepare_labels(test, key)

            target_results = {}
            for model_kind in args.probe_models:
                if task_type == "classification":
                    metrics = run_classification(train_feat, train_y, test_feat, test_y, model_kind)
                else:
                    metrics = run_regression(train_feat, train_y, test_feat, test_y, model_kind)

                target_results[model_kind] = metrics
                print(f"  target={target:20s} model={model_kind:6s} metrics={metrics}")

            feature_results["targets"][target] = target_results

        summary["results"][feature_type] = feature_results

    out_path = output_dir / "baseline_probe_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved baseline results to: {out_path}")


if __name__ == "__main__":
    main()
