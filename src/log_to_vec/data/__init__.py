"""Data module for log parsing and preprocessing."""

try:
    from .log_parser import LogParser
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local data stack
    if exc.name != "pandas":
        raise
    LogParser = None

try:
    from .preprocessor import LogPreprocessor, create_sequences
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local data stack
    if exc.name != "pandas":
        raise
    LogPreprocessor = None
    create_sequences = None

try:
    from .dataset import LogDataset
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local ML stack
    if exc.name not in {"pandas", "torch"}:
        raise
    LogDataset = None

__all__ = ["LogParser", "LogDataset", "LogPreprocessor", "create_sequences"]
