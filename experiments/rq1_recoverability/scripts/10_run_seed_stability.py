"""Run a small multi-seed MOMENT stability sweep and summarize recoverability metrics."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
import subprocess
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the end-to-end RQ1 MOMENT pipeline over multiple seeds and summarize stability."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["frs_clean_vnext_long", "frs_noisy_vnext_long"],
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 43, 44],
        help="Seed list for full end-to-end runs.",
    )
    parser.add_argument("--run_prefix", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python_exe", type=str, default=str(Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"))
    parser.add_argument("--num_trajectories", type=int, default=120)
    parser.add_argument("--trajectory_length", type=int, default=320)
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--num_channels", type=int, default=4)
    parser.add_argument("--window_length", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--moment_per_sample_standardize", action="store_true")
    return parser.parse_args()


def run_step(command: list[str], cwd: Path) -> None:
    print("\nRunning:", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(summary: dict, path: Iterable[str]) -> float:
    node = summary
    for key in path:
        node = node[key]
    return float(node)


def summarize_metric(values: list[float]) -> dict:
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
        "values": values,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    repo_root = root.parents[1]
    scripts_dir = root / "scripts"

    run_prefix = args.run_prefix or f"{args.dataset_name}_seed_stability"
    summary_rows: list[dict[str, object]] = []

    for seed in args.seeds:
        run_name = f"{run_prefix}__seed{seed}"
        command = [
            args.python_exe,
            str(scripts_dir / "07_run_frs_moment_recoverability.py"),
            "--dataset_name",
            args.dataset_name,
            "--run_name",
            run_name,
            "--seed",
            str(seed),
            "--num_trajectories",
            str(args.num_trajectories),
            "--trajectory_length",
            str(args.trajectory_length),
            "--latent_dim",
            str(args.latent_dim),
            "--num_channels",
            str(args.num_channels),
            "--window_length",
            str(args.window_length),
            "--stride",
            str(args.stride),
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
            "--python_exe",
            args.python_exe,
        ]
        if args.moment_per_sample_standardize:
            command.append("--moment_per_sample_standardize")
        run_step(command, cwd=repo_root)

        run_root = root / "artifacts" / "runs" / run_name
        summary_payload = load_json(run_root / "moment_pretrained" / "evaluation" / "recoverability_summary.json")
        baseline_payload = load_json(run_root / "baselines" / "baseline_probe_results.json")

        row = {
            "seed": seed,
            "run_name": run_name,
            "mode_rbf": metric(summary_payload, ["probes", "mode_id", "rbf", "balanced_accuracy"]),
            "spectral_rbf": metric(summary_payload, ["probes", "spectral_id", "rbf", "balanced_accuracy"]),
            "coupling_rbf": metric(summary_payload, ["probes", "coupling_id", "rbf", "balanced_accuracy"]),
            "transition_rbf": metric(summary_payload, ["probes", "is_transition_window", "rbf", "balanced_accuracy"]),
            "mean_load_r2": metric(summary_payload, ["probes", "mean_load", "rbf", "r2"]),
            "retrieval_mode_p10": metric(summary_payload, ["retrieval", "mode_id", "precision_at_10"]),
            "retrieval_spectral_p10": metric(summary_payload, ["retrieval", "spectral_id", "precision_at_10"]),
            "retrieval_coupling_p10": metric(summary_payload, ["retrieval", "coupling_id", "precision_at_10"]),
            "cluster_mode_ari": metric(summary_payload, ["clustering", "mode_id", "reference_run", "ari"]),
            "cluster_spectral_ari": metric(summary_payload, ["clustering", "spectral_id", "reference_run", "ari"]),
            "cluster_coupling_ari": metric(summary_payload, ["clustering", "coupling_id", "reference_run", "ari"]),
            "baseline_summary_mode": float(
                baseline_payload["results"]["summary"]["targets"]["mode_id"]["linear"]["balanced_accuracy"]
            ),
            "baseline_summary_spectral": float(
                baseline_payload["results"]["summary"]["targets"]["spectral_id"]["linear"]["balanced_accuracy"]
            ),
            "baseline_summary_coupling": float(
                baseline_payload["results"]["summary"]["targets"]["coupling_id"]["linear"]["balanced_accuracy"]
            ),
            "baseline_summary_mean_load_r2": float(
                baseline_payload["results"]["summary"]["targets"]["mean_load"]["linear"]["r2"]
            ),
        }
        summary_rows.append(row)

    aggregate = {
        "mode_rbf": summarize_metric([float(r["mode_rbf"]) for r in summary_rows]),
        "spectral_rbf": summarize_metric([float(r["spectral_rbf"]) for r in summary_rows]),
        "coupling_rbf": summarize_metric([float(r["coupling_rbf"]) for r in summary_rows]),
        "transition_rbf": summarize_metric([float(r["transition_rbf"]) for r in summary_rows]),
        "mean_load_r2": summarize_metric([float(r["mean_load_r2"]) for r in summary_rows]),
        "retrieval_mode_p10": summarize_metric([float(r["retrieval_mode_p10"]) for r in summary_rows]),
        "retrieval_spectral_p10": summarize_metric([float(r["retrieval_spectral_p10"]) for r in summary_rows]),
        "retrieval_coupling_p10": summarize_metric([float(r["retrieval_coupling_p10"]) for r in summary_rows]),
        "cluster_mode_ari": summarize_metric([float(r["cluster_mode_ari"]) for r in summary_rows]),
        "cluster_spectral_ari": summarize_metric([float(r["cluster_spectral_ari"]) for r in summary_rows]),
        "cluster_coupling_ari": summarize_metric([float(r["cluster_coupling_ari"]) for r in summary_rows]),
    }

    output_dir = root / "artifacts" / "runs" / run_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "seed_stability_summary.json"
    payload = {
        "dataset_name": args.dataset_name,
        "seeds": args.seeds,
        "moment_per_sample_standardize": args.moment_per_sample_standardize,
        "rows": summary_rows,
        "aggregate": aggregate,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSaved seed stability summary to:", output_path)
    for key, value in aggregate.items():
        print(f"{key}: mean={value['mean']:.4f}, std={value['std']:.4f}")


if __name__ == "__main__":
    main()
