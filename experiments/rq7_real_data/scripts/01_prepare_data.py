"""Step 1: Load real-world datasets and produce windows.npz in rq4-compatible format.

Output: results/data/{scenario_id}/windows.npz
Fields:
  X              [N, L, C] float32  — sliding windows
  mode_id        [N]       int32    — label (0=normal, 1=anomaly, or mode index)
  run_id         [N]       int32    — pseudo-run or real run index
  window_start   [N]       int64    — position of window in the source time series
  pseudo_runs    scalar    bool     — stored as int8 (1=pseudo, 0=genuine)

Usage:
  python 01_prepare_data.py --scenario swat_binary
  python 01_prepare_data.py --scenario msl_binary
  python 01_prepare_data.py --scenario tep_modes    # requires TEP download
  python 01_prepare_data.py --scenario tep_faults
  python 01_prepare_data.py                         # all available scenarios
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE    = Path(__file__).resolve().parent
_RQ7_SRC = _HERE.parent / "src"
if str(_RQ7_SRC) not in sys.path:
    sys.path.insert(0, str(_RQ7_SRC))

from rq7.config.scenarios import SCENARIO_IDS, get_scenario

WINDOW_LEN = 128
STRIDE     = 16


def _sliding_windows(X: np.ndarray, mode_id: np.ndarray, run_id: np.ndarray,
                     win: int, stride: int):
    """Slice [T,C] signal into [N,L,C] windows respecting run boundaries."""
    all_X, all_mode, all_run, all_start = [], [], [], []

    for rid in np.unique(run_id):
        mask  = run_id == rid
        idx   = np.where(mask)[0]
        # must be contiguous (they always are for our loaders)
        x_r   = X[idx]
        m_r   = mode_id[idx]
        T_r   = len(x_r)
        for s in range(0, T_r - win + 1, stride):
            all_X.append(x_r[s:s+win])
            all_mode.append(m_r[s + win // 2])   # label at window centre
            all_run.append(rid)
            all_start.append(idx[s])

    return (
        np.stack(all_X).astype(np.float32),
        np.array(all_mode, dtype=np.int32),
        np.array(all_run,  dtype=np.int32),
        np.array(all_start, dtype=np.int64),
    )


def prepare_scenario(sid: str, out_dir: Path, force: bool) -> None:
    sc = get_scenario(sid)
    out_path = out_dir / sid / "windows.npz"
    if out_path.exists() and not force:
        print(f"  [skip] {sid}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamic import of the loader
    if sc.dataset == "SWaT":
        from rq7.data.swat_loader import load_swat_binary as loader_fn
    elif sc.dataset == "MSL":
        from rq7.data.msl_loader import load_msl_binary as loader_fn
    elif sc.dataset == "TEP" and sc.id == "tep_modes":
        from rq7.data.tep_loader import load_tep_modes as loader_fn
    elif sc.dataset == "TEP" and sc.id == "tep_faults":
        from rq7.data.tep_loader import load_tep_faults as loader_fn
    else:
        raise ValueError(f"No loader mapped for scenario {sid!r}")

    print(f"  Loading {sid} …", end=" ", flush=True)
    data = loader_fn()
    X, mode_id, run_id = data["X"], data["mode_id"], data["run_id"]
    print(f"raw shape {X.shape}  modes {np.unique(mode_id).tolist()}")

    X_w, mode_w, run_w, start_w = _sliding_windows(
        X, mode_id, run_id, WINDOW_LEN, STRIDE
    )
    print(f"    → windows {X_w.shape}  "
          f"mode dist {np.unique(mode_w, return_counts=True)}")

    np.savez_compressed(
        out_path,
        X            = X_w,
        mode_id      = mode_w,
        run_id       = run_w,
        window_start = start_w,
        pseudo_runs  = np.int8(sc.pseudo_runs),
    )
    print(f"  [done] {sid} → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario",   "--scenarios", dest="scenarios",
                   nargs="+", default=None,
                   help="Scenario IDs to prepare (default: all available)")
    p.add_argument("--output_dir", default="../results/data")
    p.add_argument("--force",      action="store_true")
    return p.parse_args()


def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    ids     = args.scenarios or SCENARIO_IDS

    for sid in ids:
        try:
            prepare_scenario(sid, out_dir, args.force)
        except Exception as e:
            print(f"  [err ] {sid}: {e}")


if __name__ == "__main__":
    main()
