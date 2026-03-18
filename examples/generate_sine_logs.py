import os
import json
import math
import argparse
import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def generate_sine_block(
    freq: float,
    num_points: int,
    sampling_rate: float,
    amplitude: float = 1.0,
    phase: float = 0.0,
    component: str = "sensor_A",
    severity: str = "INFO",
    class_id: int = 0,
    start_time: int = 0
):
    """
    Generate one continuous block of sine-wave logs.

    Each row is one log event.
    The sine signal is stored in data["value"].
    """
    rows = []

    for i in range(num_points):
        t = i / sampling_rate
        value = amplitude * math.sin(2.0 * math.pi * freq * t + phase)

        # Optional extra numeric fields
        temperature = 20.0 + 0.5 * value
        pressure = 1.0 + 0.2 * value
        position = float(i)

        timestamp = start_time + i

        # Important:
        # message follows "in XXX state" pattern so preprocessor can parse state
        message = f"Process in RUN state | class={class_id} | freq={freq:.2f}"

        data_dict = {
            "temperature": temperature,
            "pressure": pressure,
            "position": position,
            "value": value,
            "threshold_exceeded": bool(abs(value) > 0.8),
            "state": "MOVING"
        }

        row = {
            "timestamp": timestamp,
            "event_type": f"SINE_CLASS_{class_id}",
            "component": component,
            "severity": severity,
            "message": message,
            "data": json.dumps(data_dict)
        }
        rows.append(row)

    return rows


def generate_sine_logs(
    frequencies,
    blocks_per_class: int,
    points_per_block: int,
    sampling_rate: float,
    amplitude: float = 1.0,
    output_path: str = "data/raw/sine_logs.csv"
):
    """
    Generate a raw CSV log dataset for the existing preprocess pipeline.

    The output is a long event stream.
    Each class corresponds to one sine frequency.
    """
    all_rows = []
    current_time = 0

    for class_id, freq in enumerate(frequencies):
        for block_idx in range(blocks_per_class):
            rows = generate_sine_block(
                freq=freq,
                num_points=points_per_block,
                sampling_rate=sampling_rate,
                amplitude=amplitude,
                phase=0.0,
                component="sensor_A",
                severity="INFO",
                class_id=class_id,
                start_time=current_time
            )
            all_rows.extend(rows)

            # Leave a small gap between blocks
            current_time += points_per_block + 5

    df = pd.DataFrame(all_rows)

    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)

    print(f"Saved raw sine logs to: {output_path}")
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    print("\nClass distribution:")
    print(df["event_type"].value_counts().sort_index())

    return df


def main(args):
    frequencies = [float(x) for x in args.frequencies.split(",")]

    generate_sine_logs(
        frequencies=frequencies,
        blocks_per_class=args.blocks_per_class,
        points_per_block=args.points_per_block,
        sampling_rate=args.sampling_rate,
        amplitude=args.amplitude,
        output_path=args.output
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--frequencies",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated sine frequencies"
    )
    parser.add_argument(
        "--blocks_per_class",
        type=int,
        default=20,
        help="Number of blocks per frequency class"
    )
    parser.add_argument(
        "--points_per_block",
        type=int,
        default=100,
        help="Number of log rows in each block"
    )
    parser.add_argument(
        "--sampling_rate",
        type=float,
        default=50.0,
        help="Sampling rate in Hz"
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/sine_logs.csv"
    )

    args = parser.parse_args()
    main(args)