"""
1. Read raw csv
2. Call LogPreprocessor
3. Generate feature matrix
4. Call create_sequences
5. Generate sliding windows
6. Split train/val/test
7. Save dataset

"""

import os
import json
import argparse
import numpy as np
import pandas as pd

from preprocessor import LogPreprocessor, create_sequences


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_sequences(sequences, train_ratio, val_ratio, test_ratio, split_mode="sequential", seed=42
):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    n = len(sequences)

    if split_mode not in ["sequential", "random"]:
        raise ValueError("split_mode must be 'sequential' or 'random'")

    if split_mode == "random":
        rng = np.random.default_rng(seed)
        indices = np.arange(n)
        rng.shuffle(indices)
        sequences = sequences[indices]

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = sequences[:n_train]
    val = sequences[n_train:n_train + n_val]
    test = sequences[n_train + n_val:]

    return train, val, test


def save_split(path, X):
    np.savez_compressed(path, X=X.astype(np.float32))


def main(args):

    ensure_dir(args.output_dir)

    print("\nLoading raw dataset...")
    logs_df = pd.read_csv(args.input)

    print("Rows:", len(logs_df))
    print("Columns:", logs_df.columns.tolist())

    print("\nRunning LogPreprocessor...")
    preprocessor = LogPreprocessor()

    feature_matrix = preprocessor.fit_transform(
        logs_df,
        normalize=args.normalize
    )

    print("Feature matrix shape:", feature_matrix.shape)

    print("\nCreating sequences...")
    sequences = create_sequences(
        feature_matrix,
        sequence_length=args.sequence_length,
        stride=args.stride
    )

    print("Sequence shape:", sequences.shape)

    print("\nSplitting dataset...")
    train, val, test = split_sequences(
        sequences,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        split_mode=args.split_mode,
        seed=args.seed
    )

    print("Train:", train.shape)
    print("Val:", val.shape)
    print("Test:", test.shape)

    print("\nSaving processed dataset...")

    save_split(os.path.join(args.output_dir, "train.npz"), train)
    save_split(os.path.join(args.output_dir, "val.npz"), val)
    save_split(os.path.join(args.output_dir, "test.npz"), test)

    preprocessor.save(os.path.join(args.output_dir, "preprocessor.json"))

    meta = {
        "input_file": args.input,
        "num_logs": int(len(logs_df)),
        "num_features": int(feature_matrix.shape[1]),
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "num_sequences": int(len(sequences)),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio
    }

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str,
                        default="data/raw/toy_logs.csv")

    parser.add_argument("--output_dir", type=str,
                        default="data/processed/version1")

    parser.add_argument("--sequence_length", type=int,
                        default=10)

    parser.add_argument("--stride", type=int,
                        default=5)

    parser.add_argument("--split_mode", type=str,
                        default="sequential",
                        choices=["sequential", "random"])

    parser.add_argument("--seed", type=int,
                        default=42)

    parser.add_argument("--train_ratio", type=float,
                        default=0.7)

    parser.add_argument("--val_ratio", type=float,
                        default=0.15)

    parser.add_argument("--test_ratio", type=float,
                        default=0.15)

    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()

    main(args)