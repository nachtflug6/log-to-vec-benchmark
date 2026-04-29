"""Step 2: Extract FFT and MOMENT embeddings for all problems.

TS2Vec embeddings are handled in script 03 (requires training).

Usage:
  python 02_extract_embeddings.py --data_dir data --output_dir embeddings --model fft
  python 02_extract_embeddings.py --data_dir data --output_dir embeddings --model moment --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

# Reuse FFT baseline from the existing benchmark infrastructure
_repo_src = Path(__file__).parent.parent.parent.parent / "src"
sys.path.insert(0, str(_repo_src))
from version2.evaluation.baseline_features import fft_features, summary_stat_features

PROBLEMS = ["p1_simple_1d", "p2_multichannel", "p3_hard_noisy"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--output_dir", type=str, default="embeddings")
    p.add_argument("--model", type=str, choices=["fft", "summary", "moment"], default="fft")
    p.add_argument("--problems", nargs="+", default=PROBLEMS)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--moment_model_name", type=str, default="AutonLab/MOMENT-1-base")
    p.add_argument("--fft_keep_bins", type=int, default=8)
    return p.parse_args()


# ---------------------------------------------------------------------------
# MOMENT extractor (reuses pattern from examples/moment/extract_moment_embeddings.py)
# ---------------------------------------------------------------------------

def _load_moment(model_name: str, device: torch.device):
    try:
        from momentfm import MOMENTPipeline
    except ImportError as e:
        raise ImportError("momentfm is not installed.") from e

    model = MOMENTPipeline.from_pretrained(
        model_name, model_kwargs={"task_name": "embedding"}
    )
    try:
        model.init()
    except AttributeError:
        pass
    model.to(device)
    model.eval()
    return model


def _extract_moment(X: np.ndarray, model, device: torch.device, batch_size: int) -> np.ndarray:
    """X: [N, L, C] -> embeddings [N, D]."""
    N, L, C = X.shape
    all_emb: List[np.ndarray] = []

    for start in range(0, N, batch_size):
        batch = X[start : start + batch_size]  # [B, L, C]
        # MOMENT expects [B, C, L] (channel-first)
        t = torch.from_numpy(batch.transpose(0, 2, 1)).float().to(device)
        with torch.no_grad():
            out = model(x_enc=t)
        if hasattr(out, "embeddings"):
            emb = out.embeddings.cpu().numpy()
        else:
            emb = out.cpu().numpy()
        all_emb.append(emb)

    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def embed(X: np.ndarray, model_name: str, args: argparse.Namespace) -> np.ndarray:
    """Dispatch to the correct embedding function."""
    if model_name == "fft":
        # Pad single-channel arrays to 3D if needed
        if X.ndim == 2:
            X = X[:, :, None]
        return fft_features(X, keep_bins=args.fft_keep_bins)

    if model_name == "summary":
        if X.ndim == 2:
            X = X[:, :, None]
        return summary_stat_features(X)

    if model_name == "moment":
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        model = _load_moment(args.moment_model_name, device)
        if X.ndim == 2:
            X = X[:, :, None]
        return _extract_moment(X, model, device, args.batch_size)

    raise ValueError(f"Unknown model: {model_name}")


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    for problem in args.problems:
        problem_dir = data_root / problem
        windows_path = problem_dir / "windows.npz"
        if not windows_path.exists():
            print(f"[skip] {problem}: windows.npz not found at {windows_path}")
            continue

        data = np.load(windows_path)
        X = data["X"]  # [N, L, C]

        print(f"[{problem}] Extracting {args.model} embeddings from {X.shape} windows...")
        emb = embed(X, args.model, args)

        out_path = out_root / f"{problem}_{args.model}.npz"
        np.savez_compressed(
            out_path,
            embeddings=emb,
            mode_id=data["mode_id"],
            trajectory_id=data["trajectory_id"],
            window_start=data["window_start"],
            is_transition_window=data["is_transition_window"],
            distance_to_boundary=data["distance_to_boundary"],
        )
        print(f"  -> {out_path}  shape={emb.shape}")


if __name__ == "__main__":
    main()
