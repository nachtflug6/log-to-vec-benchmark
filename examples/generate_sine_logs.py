import os
import json
import math
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sympy.core.random import rng


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


COMPLEXITY_PRESETS = {
    "simple": {
        "amplitude_range": (0.9, 1.1),
        "phase_random": True,
        "noise_std_range": (0.01, 0.03),
        "drift_std_range": (0.0, 0.0),
        "spike_rate": 0.0,
        "spike_magnitude_range": (0.0, 0.0),
        "missing_rate": 0.0,
        "use_state_segments": False,
        "use_multi_frequency": False,
        "secondary_amp_range": (0.0, 0.0),
        "temperature_noise_std": 0.05,
        "pressure_noise_std": 0.02,
        "position_noise_std": 0.05,
        "threshold_base": 0.85,
    },
    "medium": {
        "amplitude_range": (0.8, 1.2),
        "phase_random": True,
        "noise_std_range": (0.03, 0.07),
        "drift_std_range": (0.005, 0.02),
        "spike_rate": 0.01,
        "spike_magnitude_range": (0.3, 0.7),
        "missing_rate": 0.01,
        "use_state_segments": True,
        "use_multi_frequency": False,
        "secondary_amp_range": (0.0, 0.0),
        "temperature_noise_std": 0.10,
        "pressure_noise_std": 0.04,
        "position_noise_std": 0.10,
        "threshold_base": 0.80,
    },
    "complex": {
        "amplitude_range": (0.7, 1.3),
        "phase_random": True,
        "noise_std_range": (0.05, 0.12),
        "drift_std_range": (0.01, 0.05),
        "spike_rate": 0.03,
        "spike_magnitude_range": (0.5, 1.2),
        "missing_rate": 0.03,
        "use_state_segments": True,
        "use_multi_frequency": True,
        "secondary_amp_range": (0.10, 0.35),
        "temperature_noise_std": 0.15,
        "pressure_noise_std": 0.06,
        "position_noise_std": 0.15,
        "threshold_base": 0.75,
    },
}


MESSAGE_TEMPLATES = {
    "STARTUP": [
        "Process in STARTUP state | class={class_id} | freq={freq:.2f}",
        "System entering STARTUP state | class={class_id} | main_freq={freq:.2f}",
        "STARTUP phase active | class={class_id} | frequency={freq:.2f}",
    ],
    "RUN": [
        "Process in RUN state | class={class_id} | freq={freq:.2f}",
        "System running in RUN state | class={class_id} | main_freq={freq:.2f}",
        "RUN phase active | class={class_id} | frequency={freq:.2f}",
    ],
    "COOLING": [
        "Process in COOLING state | class={class_id} | freq={freq:.2f}",
        "System entering COOLING state | class={class_id} | main_freq={freq:.2f}",
        "COOLING phase active | class={class_id} | frequency={freq:.2f}",
    ],
    "ERROR": [
        "Process in ERROR state | class={class_id} | freq={freq:.2f}",
        "System fault in ERROR state | class={class_id} | main_freq={freq:.2f}",
        "ERROR phase active | class={class_id} | frequency={freq:.2f}",
    ],
}


COMPONENT_POOL = [
    "sensor_A",
    "sensor_B",
    "actuator_X",
    "controller_Y",
]


ACTUATOR_STATE_BY_LOG_STATE = {
    "STARTUP": ["MOVING", "OPEN"],
    "RUN": ["MOVING", "OPEN", "CLOSED"],
    "COOLING": ["MOVING", "CLOSED"],
    "ERROR": ["MOVING", "OPEN", "CLOSED"],
}


def sample_block_config(
    rng: np.random.Generator,
    complexity: str,
    main_freq: float,
    anomaly_rate: float,
    anomaly_strength: float,
) -> Dict:
    preset = COMPLEXITY_PRESETS[complexity]

    amplitude = rng.uniform(*preset["amplitude_range"])
    phase = rng.uniform(0.0, 2.0 * math.pi) if preset["phase_random"] else 0.0
    noise_std = rng.uniform(*preset["noise_std_range"])
    drift_std = rng.uniform(*preset["drift_std_range"])

    has_error = rng.random() < anomaly_rate

    secondary_freq = None
    secondary_amp = 0.0
    if preset["use_multi_frequency"]:
        secondary_freq = main_freq * rng.choice([0.5, 1.5, 2.0])
        secondary_amp = rng.uniform(*preset["secondary_amp_range"])

    component = rng.choice(COMPONENT_POOL)

    return {
        "amplitude": float(amplitude),
        "phase": float(phase),
        "noise_std": float(noise_std),
        "drift_std": float(drift_std),
        "component": str(component),
        "has_error": bool(has_error),
        "anomaly_strength": float(anomaly_strength),
        "secondary_freq": float(secondary_freq) if secondary_freq is not None else None,
        "secondary_amp": float(secondary_amp),
    }


def build_state_schedule(
    num_points: int,
    use_state_segments: bool,
    has_error: bool,
    error_segment_ratio: float,
) -> List[str]:
    if not use_state_segments:
        states = ["RUN"] * num_points
    else:
        n_startup = max(1, int(num_points * 0.2))
        n_cooling = max(1, int(num_points * 0.2))
        n_run = max(1, num_points - n_startup - n_cooling)

        states = (
            ["STARTUP"] * n_startup
            + ["RUN"] * n_run
            + ["COOLING"] * n_cooling
        )

        states = states[:num_points]

    if has_error:
        error_len = max(1, int(num_points * error_segment_ratio))
        center_low = max(0, int(num_points * 0.35))
        center_high = max(center_low + 1, int(num_points * 0.75))
        start_idx = rng.integers(center_low, center_high)
        end_idx = min(num_points, start_idx + error_len)
        for i in range(start_idx, end_idx):
            states[i] = "ERROR"

    return states


def state_envelope(state: str, local_ratio: float) -> float:
    if state == "STARTUP":
        return 0.3 + 0.7 * local_ratio
    if state == "RUN":
        return 1.0
    if state == "COOLING":
        return 1.0 - 0.6 * local_ratio
    if state == "ERROR":
        return 1.1
    return 1.0


def state_bias(state: str) -> float:
    if state == "STARTUP":
        return 0.05
    if state == "RUN":
        return 0.0
    if state == "COOLING":
        return -0.05
    if state == "ERROR":
        return 0.15
    return 0.0


def choose_severity(state: str, threshold_exceeded: bool, abs_value: float) -> str:
    if state == "ERROR":
        return "ERROR"
    if threshold_exceeded or abs_value > 0.9:
        return "WARNING"
    return "INFO"


def choose_actuator_state(
    rng: np.random.Generator,
    log_state: str
) -> str:
    candidates = ACTUATOR_STATE_BY_LOG_STATE.get(log_state, ["MOVING"])
    return str(rng.choice(candidates))


def build_message(
    rng: np.random.Generator,
    state: str,
    class_id: int,
    freq: float
) -> str:
    template = rng.choice(MESSAGE_TEMPLATES[state])
    return template.format(class_id=class_id, freq=freq)


def maybe_nan(rng: np.random.Generator, value: float, missing_rate: float):
    if rng.random() < missing_rate:
        return None
    return float(value)


def generate_sine_block(
    rng: np.random.Generator,
    freq: float,
    class_id: int,
    block_id: int,
    num_points: int,
    sampling_rate: float,
    complexity: str,
    start_time: int,
    anomaly_rate: float,
    anomaly_strength: float,
    error_segment_ratio: float,
) -> List[Dict]:
    preset = COMPLEXITY_PRESETS[complexity]
    config = sample_block_config(
        rng=rng,
        complexity=complexity,
        main_freq=freq,
        anomaly_rate=anomaly_rate,
        anomaly_strength=anomaly_strength,
    )

    states = build_state_schedule(
        num_points=num_points,
        use_state_segments=preset["use_state_segments"],
        has_error=config["has_error"],
        error_segment_ratio=error_segment_ratio,
    )

    rows = []

    drift_slope = rng.normal(0.0, config["drift_std"])
    drift_offset = rng.normal(0.0, config["drift_std"])

    for i in range(num_points):
        t = i / sampling_rate
        state = states[i]

        # Segment-relative position
        same_state_indices = [j for j, s in enumerate(states) if s == state]
        if len(same_state_indices) > 1:
            local_pos = same_state_indices.index(i)
            local_ratio = local_pos / (len(same_state_indices) - 1)
        else:
            local_ratio = 0.0

        envelope = state_envelope(state, local_ratio)

        # Main sine
        value = (
            config["amplitude"]
            * envelope
            * math.sin(2.0 * math.pi * freq * t + config["phase"])
        )

        # Optional secondary sine
        if config["secondary_freq"] is not None and config["secondary_amp"] > 0.0:
            value += (
                config["secondary_amp"]
                * math.sin(
                    2.0 * math.pi * config["secondary_freq"] * t
                    + 0.5 * config["phase"]
                )
            )

        # Drift
        value += drift_offset + drift_slope * t

        # State bias
        value += state_bias(state)

        # Noise
        noise_scale = config["noise_std"]
        if state == "ERROR":
            noise_scale *= (1.5 * config["anomaly_strength"])
        value += rng.normal(0.0, noise_scale)

        # Spikes
        is_spike = False
        spike_value = 0.0
        if rng.random() < preset["spike_rate"] or (state == "ERROR" and rng.random() < 0.20):
            spike_mag = rng.uniform(*preset["spike_magnitude_range"])
            spike_sign = rng.choice([-1.0, 1.0])
            spike_value = spike_sign * spike_mag * config["anomaly_strength"]
            value += spike_value
            is_spike = True

        # Threshold logic
        threshold = preset["threshold_base"]
        if state == "ERROR":
            threshold *= 0.9
        threshold_exceeded = bool(abs(value) > threshold or is_spike)

        # Numeric fields
        temperature = (
            22.0
            + 0.25 * value
            + (0.25 if state == "STARTUP" else 0.0)
            - (0.15 if state == "COOLING" else 0.0)
            + rng.normal(0.0, preset["temperature_noise_std"])
        )

        pressure = (
            1.2
            + 0.12 * value
            + (0.10 if state == "RUN" else 0.0)
            + (0.20 if state == "ERROR" else 0.0)
            + rng.normal(0.0, preset["pressure_noise_std"])
        )

        position = (
            float(i) / max(1, num_points - 1) * 100.0
            + rng.normal(0.0, preset["position_noise_std"])
        )

        actuator_state = choose_actuator_state(rng, state)
        severity = choose_severity(state, threshold_exceeded, abs(value))
        timestamp = start_time + i

        message = build_message(rng, state, class_id, freq)

        data_dict = {
            "temperature": maybe_nan(rng, temperature, preset["missing_rate"]),
            "pressure": maybe_nan(rng, pressure, preset["missing_rate"]),
            "position": maybe_nan(rng, position, preset["missing_rate"]),
            "value": maybe_nan(rng, value, preset["missing_rate"]),
            "threshold_exceeded": threshold_exceeded,
            "state": actuator_state,
            "signal_state": state,
            "is_spike": is_spike,
            "spike_value": float(spike_value),
        }

        row = {
            "timestamp": timestamp,
            "event_type": f"SINE_CLASS_{class_id}",
            "component": config["component"],
            "severity": severity,
            "message": message,
            "data": json.dumps(data_dict),

            # Extra metadata for analysis / future block split
            "block_id": block_id,
            "class_id": class_id,
            "main_freq": float(freq),
            "amplitude": config["amplitude"],
            "phase": config["phase"],
            "noise_std": config["noise_std"],
            "drift_std": config["drift_std"],
            "complexity": complexity,
            "has_error": config["has_error"],
            "secondary_freq": config["secondary_freq"],
            "secondary_amp": config["secondary_amp"],
        }
        rows.append(row)

    return rows


def generate_sine_logs(
    frequencies: List[float],
    blocks_per_class: int,
    points_per_block: int,
    sampling_rate: float,
    complexity: str = "simple",
    anomaly_rate: float = 0.10,
    anomaly_strength: float = 1.0,
    error_segment_ratio: float = 0.10,
    seed: int = 42,
    output_path: str = "data/raw/sine_logs.csv",
) -> pd.DataFrame:
    if complexity not in COMPLEXITY_PRESETS:
        raise ValueError(f"Unknown complexity: {complexity}")

    rng = np.random.default_rng(seed)
    all_rows = []
    current_time = 0
    block_id = 0

    for class_id, freq in enumerate(frequencies):
        for _ in range(blocks_per_class):
            rows = generate_sine_block(
                rng=rng,
                freq=freq,
                class_id=class_id,
                block_id=block_id,
                num_points=points_per_block,
                sampling_rate=sampling_rate,
                complexity=complexity,
                start_time=current_time,
                anomaly_rate=anomaly_rate,
                anomaly_strength=anomaly_strength,
                error_segment_ratio=error_segment_ratio,
            )
            all_rows.extend(rows)

            current_time += points_per_block + 5
            block_id += 1

    df = pd.DataFrame(all_rows)

    ensure_dir(os.path.dirname(output_path))
    df.to_csv(output_path, index=False)

    print(f"Saved raw sine logs to: {output_path}")
    print(f"Total rows: {len(df)}")
    print("Columns:", df.columns.tolist())

    print("\nClass distribution:")
    print(df["event_type"].value_counts().sort_index())

    print("\nSeverity distribution:")
    print(df["severity"].value_counts())

    print("\nComponent distribution:")
    print(df["component"].value_counts())

    print("\nBlocks with error:")
    print(df.groupby("block_id")["has_error"].first().value_counts())

    return df


def main(args):
    frequencies = [float(x.strip()) for x in args.frequencies.split(",")]

    output_path = args.output
    if output_path is None or output_path.strip() == "":
        output_path = f"data/raw/sine_{args.complexity}.csv"

    generate_sine_logs(
        frequencies=frequencies,
        blocks_per_class=args.blocks_per_class,
        points_per_block=args.points_per_block,
        sampling_rate=args.sampling_rate,
        complexity=args.complexity,
        anomaly_rate=args.anomaly_rate,
        anomaly_strength=args.anomaly_strength,
        error_segment_ratio=args.error_segment_ratio,
        seed=args.seed,
        output_path=output_path,
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
        "--complexity",
        type=str,
        default="simple",
        choices=["simple", "medium", "complex"],
        help="Complexity preset"
    )
    parser.add_argument(
        "--anomaly_rate",
        type=float,
        default=0.10,
        help="Probability that one block contains an ERROR segment"
    )
    parser.add_argument(
        "--anomaly_strength",
        type=float,
        default=1.0,
        help="Controls anomaly severity inside ERROR segments"
    )
    parser.add_argument(
        "--error_segment_ratio",
        type=float,
        default=0.10,
        help="Fraction of points in one block replaced by ERROR state"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path. If not provided, use data/raw/sine_<complexity>.csv"
    )

    args = parser.parse_args()
    main(args)