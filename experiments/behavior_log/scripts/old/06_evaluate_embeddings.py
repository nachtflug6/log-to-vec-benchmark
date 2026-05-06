"""Stage 06: evaluate behavior recoverability from window embeddings."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from behavior_log.evaluation.window_embedding_evaluator import WindowEmbeddingEvaluator
from behavior_log.data.windowing import WindowDatasetBuilder
from behavior_log.models import PCAWindowEmbedder, load_sequence_model
from behavior_log.utils.io import load_yaml, save_json


def main() -> None:
    config_name = sys.argv[1] if len(sys.argv) > 1 else "pca"
    cfg = load_yaml(ROOT / "configs" / "evaluation" / f"{config_name}.yaml")
    data = np.load(cfg["embedding_file"], allow_pickle=True)

    evaluator = WindowEmbeddingEvaluator(
        knn_k=int(cfg["knn_k"]),
        logistic_max_iter=int(cfg["logistic_max_iter"]),
        retrieval_k=int(cfg["retrieval_k"]),
        retrieval_ks=list(cfg.get("retrieval_ks", [1, 5, 10, 20])),
    )
    result = evaluator.evaluate(
        embeddings=data["embeddings"],
        labels=data["window_behavior_id"],
        split=data["split"],
        is_transition_window=data["is_transition_window"],
    )

    if bool(cfg.get("compute_robustness", False)):
        result.metrics.update(
            compute_robustness_metrics(
                root=ROOT,
                config_name=config_name,
                cfg=cfg,
                original_embeddings=data["embeddings"],
                labels=data["window_behavior_id"],
                split=data["split"],
            )
        )

    output_dir = Path(cfg["output_dir"])
    save_json(result.metrics, output_dir / cfg["metrics_file"])
    np.savez_compressed(
        output_dir / "evaluation_artifacts.npz",
        pca_2d=result.artifacts["pca_2d"],
        tsne_2d=result.artifacts["tsne_2d"],
        cluster_assignments=result.artifacts["cluster_assignments"],
        window_behavior_id=data["window_behavior_id"],
        trajectory_id=data["trajectory_id"],
        is_transition_window=data["is_transition_window"],
        split=data["split"],
    )

    print("Evaluation metrics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value}")

def compute_robustness_metrics(
    root: Path,
    config_name: str,
    cfg: dict,
    original_embeddings: np.ndarray,
    labels: np.ndarray,
    split: np.ndarray,
) -> dict:
    model_cfg_path = cfg.get("model_config")
    if model_cfg_path is None:
        candidate = root / "configs" / "models" / f"{config_name}.yaml"
        if candidate.exists():
            model_cfg_path = candidate
        else:
            return {}

    model_cfg = load_yaml(model_cfg_path)
    window_file = model_cfg["window_file"]
    windows = WindowDatasetBuilder.load_bundle(window_file)
    model_type = str(model_cfg.get("model_type", "pca"))
    if model_type == "pca":
        model = PCAWindowEmbedder.load(Path(model_cfg["output_dir"]) / model_cfg["model_file"])
    else:
        model = load_sequence_model(
            Path(model_cfg["output_dir"]) / model_cfg["model_file"],
            model_type=model_type,
            device=str(model_cfg.get("device", "cpu")),
        )

    robustness_specs = [
        ("missing_event", apply_missing_event_corruption, {"drop_rate": float(cfg.get("robustness_missing_rate", 0.1))}),
        ("noise_event", apply_noise_event_corruption, {"replace_rate": float(cfg.get("robustness_noise_rate", 0.1))}),
        ("wrong_severity", apply_wrong_severity_corruption, {"replace_rate": float(cfg.get("robustness_wrong_severity_rate", 0.1))}),
        ("numeric_noise", apply_numeric_noise_corruption, {"noise_std": float(cfg.get("robustness_numeric_noise_std", 0.05))}),
    ]
    metrics: dict[str, float] = {}
    test_mask = split == "test"
    base_test_embeddings = original_embeddings[test_mask]
    for name, fn, kwargs in robustness_specs:
        perturbed_X, perturbed_mask = fn(windows.X, windows.mask, **kwargs)
        perturbed_embeddings = model.transform(perturbed_X, perturbed_mask)
        perturbed_test_embeddings = perturbed_embeddings[test_mask]
        metrics[f"robustness_{name}_test_mean_cosine_similarity"] = mean_cosine_similarity(
            base_test_embeddings,
            perturbed_test_embeddings,
        )
        metrics[f"robustness_{name}_test_mean_l2_distance"] = float(
            np.mean(np.linalg.norm(base_test_embeddings - perturbed_test_embeddings, axis=1))
        )
        metrics[f"robustness_{name}_retrieval_p_at_5_test"] = evaluator_retrieval_p_at_5(
            perturbed_test_embeddings,
            labels[test_mask],
        )
    return metrics


def mean_cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = left / np.clip(np.linalg.norm(left, axis=1, keepdims=True), 1e-8, None)
    right_norm = right / np.clip(np.linalg.norm(right, axis=1, keepdims=True), 1e-8, None)
    return float(np.mean(np.sum(left_norm * right_norm, axis=1)))


def evaluator_retrieval_p_at_5(embeddings: np.ndarray, labels: np.ndarray) -> float:
    return WindowEmbeddingEvaluator._retrieval_precision_at_k(embeddings, labels, 5)


def apply_missing_event_corruption(X: np.ndarray, mask: np.ndarray, drop_rate: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X_out = X.copy()
    mask_out = mask.copy()
    valid_positions = mask_out > 0
    dropped = (rng.random(mask_out.shape) < drop_rate) & valid_positions
    X_out[dropped] = 0.0
    mask_out[dropped] = 0.0
    return X_out, mask_out


def apply_noise_event_corruption(X: np.ndarray, mask: np.ndarray, replace_rate: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X_out = X.copy()
    valid_positions = mask > 0
    replace_mask = (rng.random(mask.shape) < replace_rate) & valid_positions
    if np.any(replace_mask):
        rows, cols = np.where(replace_mask)
        X_out[rows, cols, 0] = rng.integers(0, int(np.max(X[:, :, 0])) + 1, size=len(rows))
        X_out[rows, cols, 1] = rng.integers(0, int(np.max(X[:, :, 1])) + 1, size=len(rows))
    return X_out, mask.copy()


def apply_wrong_severity_corruption(X: np.ndarray, mask: np.ndarray, replace_rate: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2)
    X_out = X.copy()
    valid_positions = mask > 0
    replace_mask = (rng.random(mask.shape) < replace_rate) & valid_positions
    if np.any(replace_mask):
        rows, cols = np.where(replace_mask)
        X_out[rows, cols, 2] = rng.integers(0, int(np.max(X[:, :, 2])) + 1, size=len(rows))
    return X_out, mask.copy()


def apply_numeric_noise_corruption(X: np.ndarray, mask: np.ndarray, noise_std: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(3)
    X_out = X.copy()
    valid = mask[..., None] > 0
    noise = rng.normal(0.0, noise_std, size=X_out[:, :, 4:].shape).astype(np.float32)
    X_out[:, :, 4:] = np.clip(X_out[:, :, 4:] + noise * valid, 0.0, 1.0)
    return X_out, mask.copy()


if __name__ == "__main__":
    main()
