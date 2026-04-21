"""Run a small MOMENT configuration ablation on an existing RQ1 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run several MOMENT extraction configurations on one existing dataset split and summarize results."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        choices=[
            "frs_clean",
            "frs_noisy",
            "frs_clean_v3",
            "frs_noisy_v3",
            "frs_clean_vnext",
            "frs_noisy_vnext",
            "frs_clean_vnext_long",
            "frs_noisy_vnext_long",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_prefix", type=str, default=None)
    parser.add_argument(
        "--baseline_run_name",
        type=str,
        default=None,
        help="Optional existing run name that contains baseline_probe_results.json. If omitted, the script will try to find the most recent matching run.",
    )
    parser.add_argument("--model_name", type=str, default="AutonLab/MOMENT-1-base")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--python_exe", type=str, default=str(Path(__file__).resolve().parents[3] / "venv" / "Scripts" / "python.exe"))
    parser.add_argument(
        "--configs",
        nargs="+",
        default=["default", "standardized", "channel_last", "standardized_channel_last"],
        choices=["default", "standardized", "channel_last", "standardized_channel_last"],
        help="Named MOMENT extraction configurations to compare.",
    )
    return parser.parse_args()


def run_step(command: List[str], cwd: Path) -> None:
    print("\nRunning:", " ".join(command))
    subprocess.run(command, check=True, cwd=cwd)


def config_flags(name: str) -> Dict[str, bool]:
    return {
        "per_sample_standardize": name in {"standardized", "standardized_channel_last"},
        "channel_last_input": name in {"channel_last", "standardized_channel_last"},
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_baseline_results_path(root: Path, dataset_name: str, explicit_run_name: str | None) -> Path:
    if explicit_run_name is not None:
        candidate = root / "artifacts" / "runs" / explicit_run_name / "baselines" / "baseline_probe_results.json"
        if not candidate.exists():
            raise FileNotFoundError(
                f"Could not find baseline probe results at {candidate}. "
                "Check --baseline_run_name or rerun the standard pipeline."
            )
        return candidate

    runs_root = root / "artifacts" / "runs"
    candidates: list[Path] = []
    for child in runs_root.iterdir():
        if not child.is_dir():
            continue
        if not child.name.startswith(dataset_name):
            continue
        baseline_path = child / "baselines" / "baseline_probe_results.json"
        if baseline_path.exists():
            candidates.append(baseline_path)

    if not candidates:
        expected = runs_root / f"{dataset_name}_moment" / "baselines" / "baseline_probe_results.json"
        raise FileNotFoundError(
            f"Expected baseline probe results near {expected}, but no matching baseline run was found for dataset '{dataset_name}'. "
            "Pass --baseline_run_name explicitly or run the standard pipeline once."
        )

    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def best_baseline_score(baseline_payload: dict, target: str) -> tuple[float, str]:
    best_score = None
    best_name = None
    if target == "mean_load":
        score_key = "r2"
        for feature_set, feature_payload in baseline_payload["results"].items():
            for model_name, metrics in feature_payload["targets"][target].items():
                score = metrics[score_key]
                if best_score is None or score > best_score:
                    best_score = score
                    best_name = f"{feature_set}/{model_name}"
    else:
        for feature_set, feature_payload in baseline_payload["results"].items():
            for model_name, metrics in feature_payload["targets"][target].items():
                score = metrics.get("balanced_accuracy", metrics.get("accuracy"))
                if best_score is None or score > best_score:
                    best_score = score
                    best_name = f"{feature_set}/{model_name}"
    assert best_score is not None and best_name is not None
    return float(best_score), best_name


def build_summary_row(config_name: str, summary_payload: dict, baseline_payload: dict) -> dict:
    row: dict[str, object] = {"config": config_name}
    for target in ["mode_id", "spectral_id", "coupling_id", "is_transition_window"]:
        baseline_score, baseline_name = best_baseline_score(baseline_payload, target)
        row[f"{target}_baseline"] = baseline_score
        row[f"{target}_baseline_name"] = baseline_name
        row[f"{target}_moment_linear"] = summary_payload["probes"][target]["linear"].get(
            "balanced_accuracy",
            summary_payload["probes"][target]["linear"].get("accuracy"),
        )
        row[f"{target}_moment_rbf"] = summary_payload["probes"][target]["rbf"].get(
            "balanced_accuracy",
            summary_payload["probes"][target]["rbf"].get("accuracy"),
        )

    load_baseline, load_baseline_name = best_baseline_score(baseline_payload, "mean_load")
    row["mean_load_r2_baseline"] = load_baseline
    row["mean_load_r2_baseline_name"] = load_baseline_name
    row["mean_load_r2_moment_linear"] = summary_payload["probes"]["mean_load"]["linear"]["r2"]
    row["mean_load_r2_moment_rbf"] = summary_payload["probes"]["mean_load"]["rbf"]["r2"]

    row["retrieval_mode_p10"] = summary_payload["retrieval"]["mode_id"]["precision_at_10"]
    row["retrieval_spectral_p10"] = summary_payload["retrieval"]["spectral_id"]["precision_at_10"]
    row["retrieval_coupling_p10"] = summary_payload["retrieval"]["coupling_id"]["precision_at_10"]

    row["cluster_mode_ari"] = summary_payload["clustering"]["mode_id"]["reference_run"]["ari"]
    row["cluster_spectral_ari"] = summary_payload["clustering"]["spectral_id"]["reference_run"]["ari"]
    row["cluster_coupling_ari"] = summary_payload["clustering"]["coupling_id"]["reference_run"]["ari"]
    return row


def build_failure_row(config_name: str, error_message: str) -> dict:
    return {
        "config": config_name,
        "status": "failed",
        "error": error_message,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    repo_root = root.parents[1]
    scripts_dir = root / "scripts"
    run_prefix = args.run_prefix or f"{args.dataset_name}_moment_ablation"

    split_dir = root / "artifacts" / "datasets" / args.dataset_name / "splits" / f"trajectory_seed{args.seed}"
    train_file = split_dir / "train_windows.npz"
    val_file = split_dir / "val_windows.npz"
    test_file = split_dir / "test_windows.npz"

    if not train_file.exists() or not val_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Expected existing split bundle under {split_dir}. "
            "Run the dataset pipeline first before launching the MOMENT ablation."
        )

    baseline_results_path = resolve_baseline_results_path(
        root=root,
        dataset_name=args.dataset_name,
        explicit_run_name=args.baseline_run_name,
    )
    baseline_payload = load_json(baseline_results_path)

    comparison_rows = []

    for config_name in args.configs:
        flags = config_flags(config_name)
        run_name = f"{run_prefix}__{config_name}"
        run_root = root / "artifacts" / "runs" / run_name
        embedding_dir = run_root / "moment_pretrained" / "embeddings"
        evaluation_dir = run_root / "moment_pretrained" / "evaluation"

        export_command = [
            args.python_exe,
            str(scripts_dir / "05_export_moment_pretrained_embeddings.py"),
            "--train_file",
            str(train_file),
            "--val_file",
            str(val_file),
            "--test_file",
            str(test_file),
            "--output_dir",
            str(embedding_dir),
            "--model_name",
            args.model_name,
            "--batch_size",
            str(args.batch_size),
            "--num_workers",
            str(args.num_workers),
            "--device",
            args.device,
            "--python_exe",
            args.python_exe,
        ]
        if flags["per_sample_standardize"]:
            export_command.append("--per_sample_standardize")
        if flags["channel_last_input"]:
            export_command.append("--channel_last_input")
        try:
            run_step(export_command, cwd=repo_root)

            eval_command = [
                args.python_exe,
                str(scripts_dir / "06_evaluate_recoverability.py"),
                "--train_embeddings",
                str(embedding_dir / "train_embeddings.npz"),
                "--val_embeddings",
                str(embedding_dir / "val_embeddings.npz"),
                "--test_embeddings",
                str(embedding_dir / "test_embeddings.npz"),
                "--train_split",
                str(train_file),
                "--val_split",
                str(val_file),
                "--test_split",
                str(test_file),
                "--output_dir",
                str(evaluation_dir),
            ]
            run_step(eval_command, cwd=repo_root)

            summary_payload = load_json(evaluation_dir / "recoverability_summary.json")
            row = build_summary_row(config_name, summary_payload, baseline_payload)
            row["status"] = "ok"
            comparison_rows.append(row)
        except subprocess.CalledProcessError as exc:
            error_message = f"subprocess failed with exit code {exc.returncode}: {' '.join(exc.cmd)}"
            print(f"\nSkipping failed config '{config_name}': {error_message}")
            comparison_rows.append(build_failure_row(config_name, error_message))

    comparison_dir = root / "artifacts" / "runs" / run_prefix
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_payload = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "model_name": args.model_name,
        "baseline_results_path": str(baseline_results_path),
        "configs": args.configs,
        "rows": comparison_rows,
    }
    output_json = comparison_dir / "moment_config_comparison.json"
    output_json.write_text(json.dumps(comparison_payload, indent=2), encoding="utf-8")

    print("\nSaved MOMENT config comparison to:", output_json)
    for row in comparison_rows:
        if row.get("status") == "failed":
            print(f"{row['config']}: FAILED")
            continue
        print(
            f"{row['config']}: "
            f"spectral_rbf={row['spectral_id_moment_rbf']:.4f}, "
            f"coupling_rbf={row['coupling_id_moment_rbf']:.4f}, "
            f"mode_rbf={row['mode_id_moment_rbf']:.4f}, "
            f"mean_load_r2={row['mean_load_r2_moment_rbf']:.4f}"
        )


if __name__ == "__main__":
    main()
