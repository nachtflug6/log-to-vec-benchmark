"""Utilities for prepared trace-level datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def tokenize_sequence(sequence: str) -> list[str]:
    text = "" if sequence is None else str(sequence).strip()
    if not text:
        return []
    return text.split()
