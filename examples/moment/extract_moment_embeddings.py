from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from version2.data.fsss_dataset import FSSSWindowDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract embeddings from a pretrained MOMENT model for FSSS-compatible splits."
    )
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--val_file", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument(
        "--model_name",
        type=str,
        default="AutonLab/MOMENT-1-base",
        help="Hugging Face MOMENT model name, e.g. AutonLab/MOMENT-1-base",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    # Optional behavior knobs
    parser.add_argument(
        "--per_sample_standardize",
        action="store_true",
        help="Apply per-sample per-channel standardization before feeding to MOMENT.",
    )
    parser.add_argument(
        "--channel_last_input",
        action="store_true",
        help=(
            "Use [B, T, C] input directly. "
            "If not set, the script will feed [B, C, T]. "
            "This exists because package-level APIs may vary."
        ),
    )
    return parser.parse_args()


def load_moment_pipeline(model_name: str, device: torch.device):
    """
    Load MOMENT pretrained model in embedding mode.

    Official docs show:
        MOMENTPipeline.from_pretrained(
            'AutonLab/MOMENT-1-base',
            model_kwargs={'task_name': 'embedding'}
        )
    """
    try:
        from momentfm import MOMENTPipeline
    except ImportError as e:
        raise ImportError(
            "momentfm is not installed. Install it with:\n"
            "  pip install momentfm\n"
            "or:\n"
            "  pip install git+https://github.com/moment-timeseries-foundation-model/moment.git"
        ) from e

    model = MOMENTPipeline.from_pretrained(
        model_name,
        model_kwargs={"task_name": "embedding"},
    )

    # Some tasks in the official docs call model.init(); embedding docs do not
    # explicitly show it, but some versions of the package may still expect it.
    if hasattr(model, "init") and callable(model.init):
        try:
            model.init()
        except Exception:
            # Safe fallback: some releases may not need / support init here.
            pass

    if hasattr(model, "to"):
        model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()

    return model


def maybe_standardize_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Standardize each sample independently per channel.

    Input shape expected: [B, T, C]
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (x - mean) / std


def try_extract_from_output(outputs: Any) -> torch.Tensor:
    """
    Be robust to small API differences across package versions.

    We try several common possibilities and return [B, D].
    """
    # Case 1: direct tensor
    if torch.is_tensor(outputs):
        if outputs.ndim == 2:
            return outputs
        if outputs.ndim == 3:
            # [B, T, D] -> mean pool over time
            return outputs.mean(dim=1)
        raise ValueError(f"Unsupported tensor output shape: {tuple(outputs.shape)}")

    # Case 2: object with common attributes
    candidate_attrs = [
        "embeddings",
        "embedding",
        "sequence_embeddings",
        "sequence_embedding",
        "repr",
        "representation",
        "representations",
        "features",
        "last_hidden_state",
    ]
    for attr in candidate_attrs:
        if hasattr(outputs, attr):
            value = getattr(outputs, attr)
            if torch.is_tensor(value):
                if value.ndim == 2:
                    return value
                if value.ndim == 3:
                    return value.mean(dim=1)

    # Case 3: dict-like output
    if isinstance(outputs, dict):
        candidate_keys = [
            "embeddings",
            "embedding",
            "sequence_embeddings",
            "sequence_embedding",
            "repr",
            "representation",
            "representations",
            "features",
            "last_hidden_state",
        ]
        for key in candidate_keys:
            if key in outputs and torch.is_tensor(outputs[key]):
                value = outputs[key]
                if value.ndim == 2:
                    return value
                if value.ndim == 3:
                    return value.mean(dim=1)

    raise ValueError(
        "Could not find embeddings in MOMENT output. "
        "Please print the output object structure for your installed momentfm version."
    )


@torch.no_grad()
def encode_batch(model, x_btc: torch.Tensor, channel_last_input: bool) -> torch.Tensor:
    """
    Encode one batch and return [B, D].

    x_btc: [B, T, C]
    """
    # Different time-series libraries sometimes expect [B, C, T] instead of [B, T, C].
    model_input = x_btc if channel_last_input else x_btc.transpose(1, 2)

    # Try the most likely call patterns
    errors = []

    # Pattern A: model(x_enc=...)
    try:
        outputs = model(x_enc=model_input)
        return try_extract_from_output(outputs)
    except Exception as e:
        errors.append(f"model(x_enc=...): {repr(e)}")

    # Pattern B: model(model_input)
    try:
        outputs = model(model_input)
        return try_extract_from_output(outputs)
    except Exception as e:
        errors.append(f"model(...): {repr(e)}")

    # Pattern C: model.embed(...)
    if hasattr(model, "embed") and callable(model.embed):
        try:
            outputs = model.embed(model_input)
            return try_extract_from_output(outputs)
        except Exception as e:
            errors.append(f"model.embed(...): {repr(e)}")

    # Pattern D: model.encode(...)
    if hasattr(model, "encode") and callable(model.encode):
        try:
            outputs = model.encode(model_input)
            return try_extract_from_output(outputs)
        except Exception as e:
            errors.append(f"model.encode(...): {repr(e)}")

    joined = "\n".join(errors)
    raise RuntimeError(
        "Failed to extract embeddings from MOMENT with all attempted call patterns.\n"
        f"Tried:\n{joined}\n\n"
        "Tip: rerun with --channel_last_input if the model expects [B, T, C]. "
        "Otherwise the default path sends [B, C, T]."
    )


@torch.no_grad()
def extract_split_embeddings(
    model,
    dataloader: DataLoader,
    device: torch.device,
    per_sample_standardize: bool,
    channel_last_input: bool,
) -> np.ndarray:
    if hasattr(model, "eval"):
        model.eval()

    all_embeddings = []

    for batch in dataloader:
        x = batch["x"].to(device, non_blocking=True)  # [B, T, C]

        if per_sample_standardize:
            x = maybe_standardize_per_sample(x)

        emb = encode_batch(
            model=model,
            x_btc=x,
            channel_last_input=channel_last_input,
        )

        if emb.ndim != 2:
            raise ValueError(f"Expected [B, D] embeddings, got {tuple(emb.shape)}")

        all_embeddings.append(emb.detach().cpu())

    if len(all_embeddings) == 0:
        return np.empty((0, 0), dtype=np.float32)

    return torch.cat(all_embeddings, dim=0).numpy().astype(np.float32)


def save_embeddings(output_path: Path, embeddings: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, embeddings=embeddings)


def main() -> None:
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading splits...")
    train_ds = FSSSWindowDataset(args.train_file)
    val_ds = FSSSWindowDataset(args.val_file)
    test_ds = FSSSWindowDataset(args.test_file)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Loading MOMENT model: {args.model_name}")
    model = load_moment_pipeline(args.model_name, device)

    print("Extracting train embeddings...")
    train_emb = extract_split_embeddings(
        model=model,
        dataloader=train_loader,
        device=device,
        per_sample_standardize=args.per_sample_standardize,
        channel_last_input=args.channel_last_input,
    )

    print("Extracting val embeddings...")
    val_emb = extract_split_embeddings(
        model=model,
        dataloader=val_loader,
        device=device,
        per_sample_standardize=args.per_sample_standardize,
        channel_last_input=args.channel_last_input,
    )

    print("Extracting test embeddings...")
    test_emb = extract_split_embeddings(
        model=model,
        dataloader=test_loader,
        device=device,
        per_sample_standardize=args.per_sample_standardize,
        channel_last_input=args.channel_last_input,
    )

    print("Saving embeddings...")
    save_embeddings(output_dir / "train_embeddings.npz", train_emb)
    save_embeddings(output_dir / "val_embeddings.npz", val_emb)
    save_embeddings(output_dir / "test_embeddings.npz", test_emb)

    print("\nDone.")
    print(f"Train embeddings: {train_emb.shape} -> {output_dir / 'train_embeddings.npz'}")
    print(f"Val embeddings:   {val_emb.shape} -> {output_dir / 'val_embeddings.npz'}")
    print(f"Test embeddings:  {test_emb.shape} -> {output_dir / 'test_embeddings.npz'}")


if __name__ == "__main__":
    main()