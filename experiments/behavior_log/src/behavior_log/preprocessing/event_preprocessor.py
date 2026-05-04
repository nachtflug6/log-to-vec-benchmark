"""Preprocess raw event logs into numeric feature vectors."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class EventFeatureBundle:
    """Container for event features and aligned metadata."""

    X: np.ndarray
    trajectory_id: np.ndarray
    cycle_id: np.ndarray
    behavior_id: np.ndarray
    is_transition: np.ndarray
    event_type: np.ndarray
    component: np.ndarray
    severity: np.ndarray
    state: np.ndarray
    feature_names: np.ndarray


class EventLogPreprocessor:
    """Fit deterministic encoders and transform raw logs."""

    def __init__(self) -> None:
        self.event_type_vocab: Dict[str, int] = {}
        self.component_vocab: Dict[str, int] = {}
        self.severity_vocab: Dict[str, int] = {}
        self.state_vocab: Dict[str, int] = {}
        self.data_feature_names: List[str] = []
        self.numeric_feature_stats: Dict[str, Dict[str, float]] = {}
        self.feature_names: List[str] = []
        self.fitted = False

    def fit(self, logs_df: pd.DataFrame) -> "EventLogPreprocessor":
        """Learn vocabularies from the raw logs."""

        self.event_type_vocab = self._build_vocab(logs_df["event_type"])
        self.component_vocab = self._build_vocab(logs_df["component"])
        self.severity_vocab = self._build_vocab(logs_df["severity"])
        self.state_vocab = self._build_vocab(logs_df["state"])
        self.data_feature_names = self._build_data_feature_names(logs_df)
        self.numeric_feature_stats = self._build_numeric_feature_stats(logs_df, self.data_feature_names)

        self.feature_names = [
            "event_type_id",
            "component_id",
            "severity_id",
            "state_id",
            "sensor_value",
            "control_value",
        ]
        self.fitted = True
        return self

    def transform(self, logs_df: pd.DataFrame) -> EventFeatureBundle:
        """Transform raw logs into event-level vectors."""

        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform().")

        df = logs_df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
        df = df.sort_values(["trajectory_id", "timestamp"]).reset_index(drop=True)

        for feature_name in self.data_feature_names:
            if feature_name not in df.columns:
                df[feature_name] = 0.0
            df[feature_name] = pd.to_numeric(df[feature_name], errors="coerce").fillna(0.0)
        if "cycle_id" not in df.columns:
            if "segment_id" in df.columns:
                df["cycle_id"] = pd.to_numeric(df["segment_id"], errors="coerce").fillna(-1).astype(np.int64)
            else:
                df["cycle_id"] = 0

        n_rows = len(df)
        n_features = len(self.feature_names)
        X = np.zeros((n_rows, n_features), dtype=np.float32)

        X[:, 0] = np.array([self.event_type_vocab[value] for value in df["event_type"].tolist()], dtype=np.float32)
        X[:, 1] = np.array([self.component_vocab[value] for value in df["component"].tolist()], dtype=np.float32)
        X[:, 2] = np.array([self.severity_vocab[value] for value in df["severity"].tolist()], dtype=np.float32)
        X[:, 3] = np.array([self.state_vocab[value] for value in df["state"].tolist()], dtype=np.float32)
        X[:, 4] = self._normalize_numeric_series(df["sensor_value"], "sensor_value")
        X[:, 5] = self._normalize_numeric_series(df["control_value"], "control_value")

        return EventFeatureBundle(
            X=X,
            trajectory_id=df["trajectory_id"].astype(np.int64).to_numpy(),
            cycle_id=df["cycle_id"].astype(np.int64).to_numpy(),
            behavior_id=df["behavior_id"].astype(np.int64).to_numpy(),
            is_transition=df["is_transition"].astype(np.int64).to_numpy(),
            event_type=df["event_type"].astype(str).to_numpy(),
            component=df["component"].astype(str).to_numpy(),
            severity=df["severity"].astype(str).to_numpy(),
            state=df["state"].astype(str).to_numpy(),
            feature_names=np.array(self.feature_names, dtype=object),
        )

    def fit_transform(self, logs_df: pd.DataFrame) -> EventFeatureBundle:
        """Fit and transform in one call."""

        return self.fit(logs_df).transform(logs_df)

    def save_state(self, path: str | Path) -> None:
        """Persist fitted state to JSON."""

        state = {
            "event_type_vocab": self.event_type_vocab,
            "component_vocab": self.component_vocab,
            "severity_vocab": self.severity_vocab,
            "state_vocab": self.state_vocab,
            "data_feature_names": self.data_feature_names,
            "numeric_feature_stats": self.numeric_feature_stats,
            "feature_names": self.feature_names,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load_state(cls, path: str | Path) -> "EventLogPreprocessor":
        """Restore a preprocessor from JSON."""

        state = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls()
        obj.event_type_vocab = {str(k): int(v) for k, v in state["event_type_vocab"].items()}
        obj.component_vocab = {str(k): int(v) for k, v in state["component_vocab"].items()}
        obj.severity_vocab = {str(k): int(v) for k, v in state["severity_vocab"].items()}
        obj.state_vocab = {str(k): int(v) for k, v in state["state_vocab"].items()}
        obj.data_feature_names = list(state["data_feature_names"])
        obj.numeric_feature_stats = {
            str(k): {"min": float(v["min"]), "max": float(v["max"])}
            for k, v in state["numeric_feature_stats"].items()
        }
        obj.feature_names = list(state["feature_names"])
        obj.fitted = True
        return obj

    @staticmethod
    def save_bundle(bundle: EventFeatureBundle, path: str | Path) -> None:
        """Persist event features to NPZ."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            X=bundle.X,
            trajectory_id=bundle.trajectory_id,
            cycle_id=bundle.cycle_id,
            behavior_id=bundle.behavior_id,
            is_transition=bundle.is_transition,
            event_type=bundle.event_type,
            component=bundle.component,
            severity=bundle.severity,
            state=bundle.state,
            feature_names=bundle.feature_names,
        )

    @staticmethod
    def load_bundle(path: str | Path) -> EventFeatureBundle:
        """Load event features from NPZ."""

        data = np.load(Path(path), allow_pickle=True)
        return EventFeatureBundle(
            X=data["X"],
            trajectory_id=data["trajectory_id"],
            cycle_id=data["cycle_id"],
            behavior_id=data["behavior_id"],
            is_transition=data["is_transition"],
            event_type=data["event_type"],
            component=data["component"],
            severity=data["severity"],
            state=data["state"],
            feature_names=data["feature_names"],
        )

    @staticmethod
    def _build_vocab(values: pd.Series) -> Dict[str, int]:
        unique_values = sorted({str(value) for value in values.tolist()})
        return {value: idx for idx, value in enumerate(unique_values)}

    @staticmethod
    def _build_data_feature_names(logs_df: pd.DataFrame) -> List[str]:
        numeric_columns = []
        for column in ("sensor_value", "control_value"):
            if column in logs_df.columns:
                numeric_columns.append(column)
        return numeric_columns

    @staticmethod
    def _build_numeric_feature_stats(logs_df: pd.DataFrame, columns: List[str]) -> Dict[str, Dict[str, float]]:
        stats: Dict[str, Dict[str, float]] = {}
        for column in columns:
            values = pd.to_numeric(logs_df[column], errors="coerce").fillna(0.0)
            stats[column] = {
                "min": float(values.min()),
                "max": float(values.max()),
            }
        return stats

    def _normalize_numeric_series(self, series: pd.Series, feature_name: str) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(np.float32).to_numpy()
        stats = self.numeric_feature_stats[feature_name]
        lower = stats["min"]
        upper = stats["max"]
        if upper <= lower:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - lower) / (upper - lower)).astype(np.float32)
