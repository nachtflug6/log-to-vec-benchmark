"""Data module for log parsing and preprocessing."""

from .log_parser import LogParser
from .preprocessor import LogPreprocessor, create_sequences

try:
    from .dataset import LogDataset
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local ML stack
    if exc.name != "torch":
        raise
    LogDataset = None

__all__ = ["LogParser", "LogDataset", "LogPreprocessor", "create_sequences"]
