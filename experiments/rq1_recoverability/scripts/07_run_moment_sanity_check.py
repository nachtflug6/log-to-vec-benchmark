"""RQ1 step 7: run the full moment sanity-check pipeline end to end."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full moment-frequency sanity-check pipeline for RQ1.")
    parser.add_argument("--run_name", type=str, default="moment_freq_tcn")
    parser.add_argument("--encoder", type=str, default="tcn", choices=["tcn", "moment_pretrained", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=1024)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--noise_std", type=float, default=0.1)
    parser.add_argument("--num_bins", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
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

    dataset_root = root / "artifacts" / "datasets"
    raw_dir = dataset_root / "moment_freq" / "raw"
    split_dir = dataset_root / "moment_freq" / "splits" / "trajectory_seed42"

    run_root = root / "artifacts" / "runs" / args.run_name
    baseline_dir = run_root / "baselines"
    tcn_train_dir = run_root / "tcn" / "training"
    tcn_embedding_dir = run_root / "tcn" / "embeddings"
    tcn_evaluation_dir = run_root / "tcn" / "evaluation"
    moment_embedding_dir = run_root / "moment_pretrained" / "embeddings"
    moment_evaluation_dir = run_root / "moment_pretrained" / "evaluation"

    run_step(
        [
            args.python_exe,
            str(scripts_dir / "01_generate_moment_frequency_dataset.py"),
            "--artifact_root",
            str(dataset_root),
            "--seed",
            str(args.seed),
            "--n_samples",
            str(args.n_samples),
            "--seq_len",
            str(args.seq_len),
            "--noise_std",
            str(args.noise_std),
            "--num_bins",
            str(args.num_bins),
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

    if args.encoder in {"tcn", "both"}:
        run_step(
            [
                args.python_exe,
                str(scripts_dir / "04_train_tcn_encoder.py"),
                "--train_file",
                str(split_dir / "train_windows.npz"),
                "--val_file",
                str(split_dir / "val_windows.npz"),
                "--output_dir",
                str(tcn_train_dir),
                "--epochs",
                str(args.epochs),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
                "--seed",
                str(args.seed),
                "--python_exe",
                args.python_exe,
            ],
            cwd=repo_root,
        )

        run_step(
            [
                args.python_exe,
                str(scripts_dir / "05_export_tcn_embeddings.py"),
                "--train_file",
                str(split_dir / "train_windows.npz"),
                "--val_file",
                str(split_dir / "val_windows.npz"),
                "--test_file",
                str(split_dir / "test_windows.npz"),
                "--checkpoint",
                str(tcn_train_dir / "best_model.pt"),
                "--output_dir",
                str(tcn_embedding_dir),
                "--batch_size",
                str(args.batch_size),
                "--device",
                args.device,
                "--python_exe",
                args.python_exe,
            ],
            cwd=repo_root,
        )

        run_step(
            [
                args.python_exe,
                str(scripts_dir / "06_evaluate_recoverability.py"),
                "--train_embeddings",
                str(tcn_embedding_dir / "train_embeddings.npz"),
                "--val_embeddings",
                str(tcn_embedding_dir / "val_embeddings.npz"),
                "--test_embeddings",
                str(tcn_embedding_dir / "test_embeddings.npz"),
                "--train_split",
                str(split_dir / "train_windows.npz"),
                "--val_split",
                str(split_dir / "val_windows.npz"),
                "--test_split",
                str(split_dir / "test_windows.npz"),
                "--output_dir",
                str(tcn_evaluation_dir),
            ],
            cwd=repo_root,
        )

    if args.encoder in {"moment_pretrained", "both"}:
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

    print("\nMoment sanity-check pipeline completed.")
    print(f"Dataset:    {raw_dir}")
    print(f"Splits:     {split_dir}")
    print(f"Baselines:  {baseline_dir}")
    if args.encoder in {"tcn", "both"}:
        print(f"TCN train:  {tcn_train_dir}")
        print(f"TCN emb:    {tcn_embedding_dir}")
        print(f"TCN eval:   {tcn_evaluation_dir}")
    if args.encoder in {"moment_pretrained", "both"}:
        print(f"MOMENT emb: {moment_embedding_dir}")
        print(f"MOMENT eval:{moment_evaluation_dir}")


if __name__ == "__main__":
    main()
