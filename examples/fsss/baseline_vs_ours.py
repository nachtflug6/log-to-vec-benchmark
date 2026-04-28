from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def round_value(x: Any, n: int = 3) -> Any:
    if isinstance(x, float):
        return round(x, n)
    return x


def round_df(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(n)
    return out


def save_markdown(path: Path, title: str, sections: List[Tuple[str, pd.DataFrame]]) -> None:
    lines: List[str] = [f"# {title}", ""]
    for section_title, df in sections:
        lines.append(f"## {section_title}")
        lines.append("")
        if df.empty:
            lines.append("_No data available._")
        else:
            lines.append(df.to_markdown(index=False))
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# -----------------------------------------------------------------------------
# Baseline parsing
# -----------------------------------------------------------------------------

CLASSIFICATION_TARGETS = {
    "mode_id": "mode_id",
    "spectral_id": "spectral_id",
    "coupling_id": "coupling_id",
    "is_transition_window": "transition",
}

REGRESSION_TARGETS = {
    "mean_load": "mean_load",
}


def build_baseline_classification_table(baseline: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    results = baseline.get("results", {})
    for feature_type, feature_block in results.items():
        targets = feature_block.get("targets", {})
        for raw_target, display_target in CLASSIFICATION_TARGETS.items():
            if raw_target not in targets:
                continue

            linear = targets[raw_target].get("linear", {})
            rbf = targets[raw_target].get("rbf", {})

            rows.append({
                "Feature Type": feature_type,
                "Target": display_target,
                "Linear Acc": linear.get("accuracy"),
                "Linear Bal Acc": linear.get("balanced_accuracy"),
                "RBF Acc": rbf.get("accuracy"),
                "RBF Bal Acc": rbf.get("balanced_accuracy"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        order = {"fft": 0, "summary": 1, "raw_flatten": 2}
        df["_order"] = df["Feature Type"].map(order).fillna(999)
        df = df.sort_values(["_order", "Target"]).drop(columns="_order").reset_index(drop=True)
    return round_df(df)


def build_baseline_regression_table(baseline: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    results = baseline.get("results", {})
    for feature_type, feature_block in results.items():
        targets = feature_block.get("targets", {})
        for raw_target, display_target in REGRESSION_TARGETS.items():
            if raw_target not in targets:
                continue

            linear = targets[raw_target].get("linear", {})
            rbf = targets[raw_target].get("rbf", {})

            rows.append({
                "Feature Type": feature_type,
                "Target": display_target,
                "Model": "linear",
                "MAE": linear.get("mae"),
                "RMSE": linear.get("rmse"),
                "R2": linear.get("r2"),
            })
            rows.append({
                "Feature Type": feature_type,
                "Target": display_target,
                "Model": "rbf",
                "MAE": rbf.get("mae"),
                "RMSE": rbf.get("rmse"),
                "R2": rbf.get("r2"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        order = {"fft": 0, "summary": 1, "raw_flatten": 2}
        df["_order"] = df["Feature Type"].map(order).fillna(999)
        df = df.sort_values(["_order", "Target", "Model"]).drop(columns="_order").reset_index(drop=True)
    return round_df(df)


def build_baseline_feature_shape_table(baseline: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    results = baseline.get("results", {})
    for feature_type, feature_block in results.items():
        rows.append({
            "Feature Type": feature_type,
            "Train Shape": str(feature_block.get("train_feature_shape")),
            "Test Shape": str(feature_block.get("test_feature_shape")),
        })

    df = pd.DataFrame(rows)
    return df


# -----------------------------------------------------------------------------
# Ours parsing
# -----------------------------------------------------------------------------

def extract_ours_probe_metrics(summary: Dict[str, Any], target_name: str) -> Dict[str, Optional[float]]:
    probe_block = summary.get("probes", {}).get(target_name, {})

    linear = probe_block.get("linear", {})
    rbf = probe_block.get("rbf", {})

    out = {
        "Linear Acc": linear.get("accuracy"),
        "Linear Bal Acc": linear.get("balanced_accuracy"),
        "RBF Acc": rbf.get("accuracy"),
        "RBF Bal Acc": rbf.get("balanced_accuracy"),
        "Linear MAE": linear.get("mae"),
        "Linear RMSE": linear.get("rmse"),
        "Linear R2": linear.get("r2"),
        "RBF MAE": rbf.get("mae"),
        "RBF RMSE": rbf.get("rmse"),
        "RBF R2": rbf.get("r2"),
    }
    return out


def map_baseline_target_to_ours_target(baseline_target: str) -> str:
    mapping = {
        "mode_id": "mode_id",
        "spectral_id": "spectral_id",
        "coupling_id": "coupling_id",
        "mean_load": "mean_load",
        "is_transition_window": "is_transition_window",
    }
    return mapping[baseline_target]


# -----------------------------------------------------------------------------
# Comparison tables
# -----------------------------------------------------------------------------

def build_baseline_vs_ours_classification_table(
    baseline: Dict[str, Any],
    ours: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    baseline_results = baseline.get("results", {})
    for feature_type, feature_block in baseline_results.items():
        targets = feature_block.get("targets", {})
        for raw_target, display_target in CLASSIFICATION_TARGETS.items():
            if raw_target not in targets:
                continue

            ours_target = map_baseline_target_to_ours_target(raw_target)
            ours_metrics = extract_ours_probe_metrics(ours, ours_target)

            linear_base = targets[raw_target].get("linear", {})
            rbf_base = targets[raw_target].get("rbf", {})

            rows.append({
                "Target": display_target,
                "Baseline Feature": feature_type,
                "Baseline Linear Acc": linear_base.get("accuracy"),
                "Baseline Linear Bal Acc": linear_base.get("balanced_accuracy"),
                "Baseline RBF Acc": rbf_base.get("accuracy"),
                "Baseline RBF Bal Acc": rbf_base.get("balanced_accuracy"),
                "Ours Linear Acc": ours_metrics.get("Linear Acc"),
                "Ours Linear Bal Acc": ours_metrics.get("Linear Bal Acc"),
                "Ours RBF Acc": ours_metrics.get("RBF Acc"),
                "Ours RBF Bal Acc": ours_metrics.get("RBF Bal Acc"),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        target_order = {"mode_id": 0, "spectral_id": 1, "coupling_id": 2, "transition": 3}
        feat_order = {"fft": 0, "summary": 1, "raw_flatten": 2}
        df["_t"] = df["Target"].map(target_order).fillna(999)
        df["_f"] = df["Baseline Feature"].map(feat_order).fillna(999)
        df = df.sort_values(["_t", "_f"]).drop(columns=["_t", "_f"]).reset_index(drop=True)
    return round_df(df)


def build_baseline_vs_ours_regression_table(
    baseline: Dict[str, Any],
    ours: Dict[str, Any],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    baseline_results = baseline.get("results", {})
    ours_metrics = extract_ours_probe_metrics(ours, "mean_load")

    for feature_type, feature_block in baseline_results.items():
        targets = feature_block.get("targets", {})
        if "mean_load" not in targets:
            continue

        linear_base = targets["mean_load"].get("linear", {})
        rbf_base = targets["mean_load"].get("rbf", {})

        rows.append({
            "Baseline Feature": feature_type,
            "Baseline Linear MAE": linear_base.get("mae"),
            "Baseline Linear RMSE": linear_base.get("rmse"),
            "Baseline Linear R2": linear_base.get("r2"),
            "Baseline RBF MAE": rbf_base.get("mae"),
            "Baseline RBF RMSE": rbf_base.get("rmse"),
            "Baseline RBF R2": rbf_base.get("r2"),
            "Ours Linear MAE": ours_metrics.get("Linear MAE"),
            "Ours Linear RMSE": ours_metrics.get("Linear RMSE"),
            "Ours Linear R2": ours_metrics.get("Linear R2"),
            "Ours RBF MAE": ours_metrics.get("RBF MAE"),
            "Ours RBF RMSE": ours_metrics.get("RBF RMSE"),
            "Ours RBF R2": ours_metrics.get("RBF R2"),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        feat_order = {"fft": 0, "summary": 1, "raw_flatten": 2}
        df["_f"] = df["Baseline Feature"].map(feat_order).fillna(999)
        df = df.sort_values("_f").drop(columns="_f").reset_index(drop=True)
    return round_df(df)


def build_best_baseline_vs_ours_table(
    baseline: Dict[str, Any],
    ours: Dict[str, Any],
) -> pd.DataFrame:
    """
    Pick the best baseline result for each target:
    - classification: highest balanced accuracy among all baseline features/models
    - regression: lowest MAE among all baseline features/models
    """
    rows: List[Dict[str, Any]] = []
    baseline_results = baseline.get("results", {})

    # Classification targets
    for raw_target, display_target in CLASSIFICATION_TARGETS.items():
        best_feature = None
        best_model = None
        best_acc = None
        best_bal_acc = -1.0

        for feature_type, feature_block in baseline_results.items():
            targets = feature_block.get("targets", {})
            if raw_target not in targets:
                continue

            for model_name in ["linear", "rbf"]:
                model_metrics = targets[raw_target].get(model_name, {})
                bal_acc = model_metrics.get("balanced_accuracy")
                acc = model_metrics.get("accuracy")
                if bal_acc is None:
                    continue
                if bal_acc > best_bal_acc:
                    best_bal_acc = bal_acc
                    best_acc = acc
                    best_feature = feature_type
                    best_model = model_name

        ours_target = map_baseline_target_to_ours_target(raw_target)
        ours_probe = ours.get("probes", {}).get(ours_target, {})
        ours_best_model = None
        ours_best_acc = None
        ours_best_bal = -1.0
        for model_name in ["linear", "rbf"]:
            model_metrics = ours_probe.get(model_name, {})
            bal_acc = model_metrics.get("balanced_accuracy")
            acc = model_metrics.get("accuracy")
            if bal_acc is None:
                continue
            if bal_acc > ours_best_bal:
                ours_best_bal = bal_acc
                ours_best_acc = acc
                ours_best_model = model_name

        rows.append({
            "Target": display_target,
            "Best Baseline Feature": best_feature,
            "Best Baseline Model": best_model,
            "Best Baseline Acc": best_acc,
            "Best Baseline Bal Acc": best_bal_acc if best_bal_acc >= 0 else None,
            "Best Ours Model": ours_best_model,
            "Best Ours Acc": ours_best_acc,
            "Best Ours Bal Acc": ours_best_bal if ours_best_bal >= 0 else None,
        })

    # Regression target
    best_feature = None
    best_model = None
    best_mae = None
    best_rmse = None
    best_r2 = None

    for feature_type, feature_block in baseline_results.items():
        targets = feature_block.get("targets", {})
        if "mean_load" not in targets:
            continue

        for model_name in ["linear", "rbf"]:
            model_metrics = targets["mean_load"].get(model_name, {})
            mae = model_metrics.get("mae")
            rmse = model_metrics.get("rmse")
            r2 = model_metrics.get("r2")
            if mae is None:
                continue
            if best_mae is None or mae < best_mae:
                best_mae = mae
                best_rmse = rmse
                best_r2 = r2
                best_feature = feature_type
                best_model = model_name

    ours_probe = ours.get("probes", {}).get("mean_load", {})
    ours_best_model = None
    ours_best_mae = None
    ours_best_rmse = None
    ours_best_r2 = None
    for model_name in ["linear", "rbf"]:
        model_metrics = ours_probe.get(model_name, {})
        mae = model_metrics.get("mae")
        rmse = model_metrics.get("rmse")
        r2 = model_metrics.get("r2")
        if mae is None:
            continue
        if ours_best_mae is None or mae < ours_best_mae:
            ours_best_mae = mae
            ours_best_rmse = rmse
            ours_best_r2 = r2
            ours_best_model = model_name

    rows.append({
        "Target": "mean_load",
        "Best Baseline Feature": best_feature,
        "Best Baseline Model": best_model,
        "Best Baseline Acc": best_mae,
        "Best Baseline Bal Acc": best_r2,
        "Best Ours Model": ours_best_model,
        "Best Ours Acc": ours_best_mae,
        "Best Ours Bal Acc": ours_best_r2,
    })

    df = pd.DataFrame(rows)

    return round_df(df)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export baseline summary and baseline-vs-ours markdown tables."
    )
    parser.add_argument(
        "--baseline_json",
        type=str,
        required=True,
        help="Path to baseline_probe_results.json",
    )
    parser.add_argument(
        "--ours_json",
        type=str,
        required=True,
        help="Path to evaluation_suite_summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="markdown_reports",
        help="Directory to save markdown outputs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    baseline = load_json(args.baseline_json)
    ours = load_json(args.ours_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Baseline only
    baseline_shapes = build_baseline_feature_shape_table(baseline)
    baseline_cls = build_baseline_classification_table(baseline)
    baseline_reg = build_baseline_regression_table(baseline)

    save_markdown(
        output_dir / "baseline_summary.md",
        "Baseline Summary",
        [
            ("Feature Shapes", baseline_shapes),
            ("Classification Tasks", baseline_cls),
            ("Regression Tasks", baseline_reg),
        ],
    )

    # 2) Baseline vs ours
    compare_cls = build_baseline_vs_ours_classification_table(baseline, ours)
    compare_reg = build_baseline_vs_ours_regression_table(baseline, ours)
    compare_best = build_best_baseline_vs_ours_table(baseline, ours)

    save_markdown(
        output_dir / "baseline_vs_ours.md",
        "Baseline vs Ours",
        [
            ("Best Overall Baseline vs Ours", compare_best),
            ("Classification Comparison", compare_cls),
            ("Regression Comparison", compare_reg),
        ],
    )

    print(f"Saved: {output_dir / 'baseline_summary.md'}")
    print(f"Saved: {output_dir / 'baseline_vs_ours.md'}")


if __name__ == "__main__":
    main()