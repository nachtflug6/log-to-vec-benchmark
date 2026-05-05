"""Step 3: Extract MOMENT embeddings for all scenarios (GPU job).

Key fix vs. RQ2: input_mask is now correctly passed to model.embed() so that
both the zero-padding region (positions >= window_length) and missing-value
positions are masked out, not treated as real signal.

Run via: sbatch slurm/rq3_moment.slurm
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_RQ3_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_RQ3_SRC))

from rq3.config.scenarios import SCENARIOS, SCENARIO_IDS, get_scenario

MOMENT_SEQ_LEN = 512


def _load_moment(model_name: str, device: torch.device):
    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(model_name, model_kwargs={"task_name": "embedding"})
    model.init()
    model.eval()
    model.to(device)
    return model


def _extract_moment_with_mask(
    X_raw: np.ndarray,      # [N, L, C] — may contain NaN
    obs_mask: np.ndarray,   # [N, L, C] bool — True = observed
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Encode windows with proper input_mask covering both padding and missing positions.

    input_mask[i, t] = 1  if t < L  AND  any channel is observed at t
                      = 0  otherwise (padded region or fully-missing timestep)
    """
    N, L, C = X_raw.shape

    # Zero-fill NaN before passing to model
    X_filled = np.nan_to_num(X_raw, nan=0.0).astype(np.float32)

    # Pad / truncate to MOMENT_SEQ_LEN
    if L < MOMENT_SEQ_LEN:
        X_padded = np.zeros((N, MOMENT_SEQ_LEN, C), dtype=np.float32)
        X_padded[:, :L, :] = X_filled
        mask_padded = np.zeros((N, MOMENT_SEQ_LEN), dtype=np.int32)
        # Mark position as valid only if at least one channel is observed at that t
        mask_padded[:, :L] = obs_mask[:, :L, :].any(axis=-1).astype(np.int32)
    else:
        X_padded = X_filled[:, :MOMENT_SEQ_LEN, :]
        mask_padded = obs_mask[:, :MOMENT_SEQ_LEN, :].any(axis=-1).astype(np.int32)

    all_embs = []
    for start in range(0, N, batch_size):
        xb = torch.tensor(X_padded[start:start + batch_size]).permute(0, 2, 1).to(device)
        mb = torch.tensor(mask_padded[start:start + batch_size]).to(device)
        with torch.no_grad():
            out = model(x_enc=xb, input_mask=mb)
        all_embs.append(out.embeddings.cpu().numpy())

    return np.concatenate(all_embs, axis=0)  # [N, D]


def _process_scenario(
    scenario_id: str,
    data_dir: Path,
    output_dir: Path,
    model,
    device: torch.device,
    batch_size: int,
    force: bool,
) -> None:
    out_path = output_dir / f"{scenario_id}_moment.npz"
    if out_path.exists() and not force:
        print(f"  [skip] {scenario_id}")
        return

    npz_path = data_dir / scenario_id / "windows.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")

    data = np.load(npz_path)
    X_raw = data["X"]
    obs_mask = data["mask"]

    t0 = time.time()
    emb = _extract_moment_with_mask(X_raw, obs_mask, model, device, batch_size)
    elapsed = time.time() - t0

    meta_keys = ["mode_id", "trajectory_id", "window_start",
                 "is_transition_window", "distance_to_boundary"]
    save_dict = {"embeddings": emb.astype(np.float32)}
    for k in meta_keys:
        save_dict[k] = data[k]

    np.savez_compressed(out_path, **save_dict)
    print(f"  [done] {scenario_id:20s}  shape={emb.shape}  {elapsed:.1f}s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="../results/data")
    p.add_argument("--output_dir", type=str, default="../results/embeddings")
    p.add_argument("--scenarios", nargs="+", default=None)
    p.add_argument("--model_name", type=str, default="AutonLab/MOMENT-1-base")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenario_ids = args.scenarios if args.scenarios else SCENARIO_IDS
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"MOMENT extraction — device={device}  model={args.model_name}")
    print(f"  data:   {data_dir}")
    print(f"  output: {output_dir}")

    model = _load_moment(args.model_name, device)

    t0_total = time.time()
    for sid in scenario_ids:
        _process_scenario(sid, data_dir, output_dir, model, device, args.batch_size, args.force)

    print(f"\nAll done in {time.time() - t0_total:.1f}s")


if __name__ == "__main__":
    main()
