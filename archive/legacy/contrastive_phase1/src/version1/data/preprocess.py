import os
import json
import argparse
import numpy as np
import pandas as pd

from preprocessor import LogPreprocessor, create_sequences


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def split_items(items, train_ratio, val_ratio, test_ratio, split_mode="sequential", seed=42):
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1")

    n = len(items)
    indices = np.arange(n)

    if split_mode not in ["sequential", "random"]:
        raise ValueError("split_mode must be 'sequential' or 'random'")

    if split_mode == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    train_items = [items[i] for i in train_idx]
    val_items = [items[i] for i in val_idx]
    test_items = [items[i] for i in test_idx]

    return train_items, val_items, test_items


def segment_logs_fixed_size(logs_df, segment_size):
    segments = []
    n = len(logs_df)

    for start in range(0, n, segment_size):
        end = min(start + segment_size, n)
        seg = logs_df.iloc[start:end].reset_index(drop=True)
        if len(seg) > 0:
            segments.append(seg)

    return segments


def segment_logs_by_time_gap(logs_df, timestamp_col="timestamp", gap_threshold=10):
    if timestamp_col not in logs_df.columns:
        raise ValueError(f"timestamp column '{timestamp_col}' not found")

    df = logs_df.sort_values(timestamp_col).reset_index(drop=True)
    timestamps = df[timestamp_col].values

    segments = []
    start = 0

    for i in range(1, len(df)):
        if timestamps[i] - timestamps[i - 1] > gap_threshold:
            seg = df.iloc[start:i].reset_index(drop=True)
            if len(seg) > 0:
                segments.append(seg)
            start = i

    last_seg = df.iloc[start:].reset_index(drop=True)
    if len(last_seg) > 0:
        segments.append(last_seg)

    return segments


def transform_segments_to_sequences(preprocessor, segments, sequence_length, stride, normalize):
    all_sequences = []

    for seg_df in segments:
        if len(seg_df) < sequence_length:
            continue

        feature_matrix = preprocessor.transform(seg_df, normalize=normalize)
        sequences = create_sequences(
            feature_matrix,
            sequence_length=sequence_length,
            stride=stride
        )

        if len(sequences) > 0:
            all_sequences.append(sequences)

    if len(all_sequences) == 0:
        return np.empty((0, sequence_length, len(preprocessor.feature_names)), dtype=np.float32)

    return np.concatenate(all_sequences, axis=0).astype(np.float32)


def save_split(path, X):
    np.savez_compressed(path, X=X.astype(np.float32))


def main(args):
    ensure_dir(args.output_dir)

    print("\nLoading raw dataset...")
    logs_df = pd.read_csv(args.input)

    print("Rows:", len(logs_df))
    print("Columns:", logs_df.columns.tolist())

    # ------------------------------------------------------------------
    # Step 1: segment raw logs before fitting / windowing
    # ------------------------------------------------------------------
    print("\nSegmenting raw logs...")
    if args.segment_method == "time_gap":
        segments = segment_logs_by_time_gap(
            logs_df,
            timestamp_col=args.timestamp_col,
            gap_threshold=args.time_gap_threshold
        )
    elif args.segment_method == "fixed_size":
        segments = segment_logs_fixed_size(
            logs_df,
            segment_size=args.segment_size
        )
    else:
        raise ValueError("Unknown segment_method")

    print(f"Number of segments: {len(segments)}")

    # ------------------------------------------------------------------
    # Step 2: split segments
    # ------------------------------------------------------------------
    print("\nSplitting segments...")
    train_segments, val_segments, test_segments = split_items(
        segments,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        split_mode=args.split_mode,
        seed=args.seed
    )

    print(f"Train segments: {len(train_segments)}")
    print(f"Val segments:   {len(val_segments)}")
    print(f"Test segments:  {len(test_segments)}")

    # ------------------------------------------------------------------
    # Step 3: fit preprocessor on train only
    # ------------------------------------------------------------------
    print("\nFitting preprocessor on train segments only...")
    train_logs_df = pd.concat(train_segments, axis=0).reset_index(drop=True)

    preprocessor = LogPreprocessor()
    preprocessor.fit(train_logs_df)

    # ------------------------------------------------------------------
    # Step 4: transform each split separately, then create windows
    # ------------------------------------------------------------------
    print("\nCreating train sequences...")
    train = transform_segments_to_sequences(
        preprocessor,
        train_segments,
        sequence_length=args.sequence_length,
        stride=args.stride,
        normalize=args.normalize
    )

    print("Creating val sequences...")
    val = transform_segments_to_sequences(
        preprocessor,
        val_segments,
        sequence_length=args.sequence_length,
        stride=args.stride,
        normalize=args.normalize
    )

    print("Creating test sequences...")
    test = transform_segments_to_sequences(
        preprocessor,
        test_segments,
        sequence_length=args.sequence_length,
        stride=args.stride,
        normalize=args.normalize
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
        "num_segments": int(len(segments)),
        "train_segments": int(len(train_segments)),
        "val_segments": int(len(val_segments)),
        "test_segments": int(len(test_segments)),
        "sequence_length": args.sequence_length,
        "stride": args.stride,
        "train_sequences": int(len(train)),
        "val_sequences": int(len(val)),
        "test_sequences": int(len(test)),
        "segment_method": args.segment_method,
        "split_mode": args.split_mode,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, default="data/raw/sine_simple.csv")
    parser.add_argument("--output_dir", type=str, default="data/processed/version2")

    parser.add_argument("--sequence_length", type=int, default=10)
    parser.add_argument("--stride", type=int, default=5)

    parser.add_argument("--segment_method", type=str, default="time_gap",
                        choices=["time_gap", "fixed_size"])
    parser.add_argument("--timestamp_col", type=str, default="timestamp")
    parser.add_argument("--time_gap_threshold", type=float, default=3)
    parser.add_argument("--segment_size", type=int, default=200)

    parser.add_argument("--split_mode", type=str, default="sequential",
                        choices=["sequential", "random"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)

    parser.add_argument("--normalize", action="store_true")

    args = parser.parse_args()
    main(args)