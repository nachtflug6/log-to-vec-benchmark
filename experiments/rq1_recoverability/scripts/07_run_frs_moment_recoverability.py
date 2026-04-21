"""RQ1 step 7: run the formal FRS + MOMENT recoverability pipeline end to end."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FRS recoverability pipeline with MOMENT pretrained embeddings.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=["frs_clean_vnext_long", "frs_noisy_vnext_long"],
    )
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_trajectories", type=int, default=120)
    parser.add_argument("--trajectory_length", type=int, default=320)
    parser.add_argument("--latent_dim", type=int, default=6)
    parser.add_argument("--num_channels", type=int, default=4)
    parser.add_argument("--window_length", type=int, default=48)
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--moment_model_name", type=str, default="AutonLab/MOMENT-1-base")
    parser.add_argument("--moment_num_workers", type=int, default=0)
    parser.add_argument("--moment_per_sample_standardize", action="store_true")
    parser.add_argument("--moment_channel_last_input", action="store_true")
    parser.add_argument("--python_exe", type=str, default=str(Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"))
    return parser.parse_args()


def run_step(command: list[str], cwd: Path) -> None:
    print("\nRunning:", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    repo_root = root.parents[1]
    scripts_dir = root / "scripts"

    run_name = args.run_name or f"{args.dataset_name}_moment_pretrained"
    dataset_root = root / "artifacts" / "datasets"
    raw_dir = dataset_root / args.dataset_name / "raw"
    split_dir = dataset_root / args.dataset_name / "splits" / f"trajectory_seed{args.seed}"

    run_root = root / "artifacts" / "runs" / run_name
    baseline_dir = run_root / "baselines"
    moment_embedding_dir = run_root / "moment_pretrained" / "embeddings"
    moment_evaluation_dir = run_root / "moment_pretrained" / "evaluation"

    run_step(
        [
            args.python_exe,
            str(scripts_dir / "01_generate_frs_dataset.py"),
            "--dataset_name",
            args.dataset_name,
            "--artifact_root",
            str(dataset_root),
            "--seed",
            str(args.seed),
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
        ],
        cwd=repo_root,
    )

    run_step(
        [
            args.python_exe,
            str(scripts_dir / "02_create_dataset_splits.py"),
            "--dataset_dir",
            str(raw_dir),
            "--output_dir",
            str(split_dir),
            "--split_by",
            "trajectory",
            "--seed",
            str(args.seed),
        ],
        cwd=repo_root,
    )

    run_step(
        [
            args.python_exe,
            str(scripts_dir / "03_compute_baseline_representations.py"),
            "--train_file",
            str(split_dir / "train_windows.npz"),
            "--test_file",
            str(split_dir / "test_windows.npz"),
            "--output_dir",
            str(baseline_dir),
            "--python_exe",
            args.python_exe,
        ],
        cwd=repo_root,
    )

    moment_export_command = [
        args.python_exe,
        str(scripts_dir / "05_export_moment_pretrained_embeddings.py"),
        "--train_file",
        str(split_dir / "train_windows.npz"),
        "--val_file",
        str(split_dir / "val_windows.npz"),
        "--test_file",
        str(split_dir / "test_windows.npz"),
        "--output_dir",
        str(moment_embedding_dir),
        "--model_name",
        args.moment_model_name,
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.moment_num_workers),
        "--device",
        args.device,
        "--python_exe",
        args.python_exe,
    ]
    if args.moment_per_sample_standardize:
        moment_export_command.append("--per_sample_standardize")
    if args.moment_channel_last_input:
        moment_export_command.append("--channel_last_input")
    run_step(moment_export_command, cwd=repo_root)

    run_step(
        [
            args.python_exe,
            str(scripts_dir / "06_evaluate_recoverability.py"),
            "--train_embeddings",
            str(moment_embedding_dir / "train_embeddings.npz"),
            "--val_embeddings",
            str(moment_embedding_dir / "val_embeddings.npz"),
            "--test_embeddings",
            str(moment_embedding_dir / "test_embeddings.npz"),
            "--train_split",
            str(split_dir / "train_windows.npz"),
            "--val_split",
            str(split_dir / "val_windows.npz"),
            "--test_split",
            str(split_dir / "test_windows.npz"),
            "--output_dir",
            str(moment_evaluation_dir),
        ],
        cwd=repo_root,
    )

    print("\nFRS + MOMENT recoverability pipeline completed.")
    print(f"Dataset:     {raw_dir}")
    print(f"Splits:      {split_dir}")
    print(f"Baselines:   {baseline_dir}")
    print(f"MOMENT emb:  {moment_embedding_dir}")
    print(f"MOMENT eval: {moment_evaluation_dir}")


if __name__ == "__main__":
    main()
