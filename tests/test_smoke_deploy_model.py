import importlib.util
import json
from pathlib import Path

import numpy as np


def load_smoke_module():
    script_path = Path(__file__).parent.parent / "scripts" / "smoke_deploy_model.py"
    spec = importlib.util.spec_from_file_location("smoke_deploy_model", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_smoke_deploy_dry_run_writes_inputs_and_summary(tmp_path):
    smoke = load_smoke_module()
    output_dir = tmp_path / "alvis_smoke"
    repo_root = Path(__file__).parent.parent

    exit_code = smoke.main(
        [
            "--run-id",
            "TEST-SMOKE",
            "--output-dir",
            str(output_dir),
            "--repo-root",
            str(repo_root),
            "--num-sequences",
            "24",
            "--sequence-length",
            "6",
            "--num-features",
            "3",
            "--num-epochs",
            "2",
            "--batch-size",
            "4",
            "--dry-run",
        ]
    )

    assert exit_code == 0

    npz_path = output_dir / "data" / "smoke_sequences.npz"
    config_path = output_dir / "smoke_contrastive_config.yaml"
    summary_path = output_dir / "smoke_summary.json"

    assert npz_path.exists()
    assert config_path.exists()
    assert summary_path.exists()

    with np.load(npz_path) as npz:
        assert npz["sequences"].shape == (24, 6, 3)
        assert str(npz["run_id"]) == "TEST-SMOKE"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["data"]["npz_file"] == str(npz_path)
    assert config["training"]["num_epochs"] == 2
    assert config["logging"]["checkpoint_dir"] == str(output_dir / "checkpoints")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dry_run"] is True
    assert summary["run_id"] == "TEST-SMOKE"
    assert summary["sequence_shape"] == [24, 6, 3]
    assert summary["artifacts"] == {
        "best_model_exists": False,
        "embeddings_exists": False,
    }
