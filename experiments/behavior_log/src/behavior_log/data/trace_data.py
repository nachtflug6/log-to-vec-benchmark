"""Utilities for prepared trace-level datasets."""

from __future__ import annotations

from pathlib import Path
import json

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
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text.split()
    if isinstance(parsed, list):
        return [str(token) for token in parsed if str(token)]
    if parsed is None:
        return []
    return [str(parsed)]
