"""Log-to-Vec: Benchmarking embeddings for time-stamped log sequences."""

__version__ = "0.1.0"

try:
    from .data.log_parser import LogParser
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local data stack
    if exc.name != "pandas":
        raise
    LogParser = None

try:
    from .data.dataset import LogDataset
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local ML stack
    if exc.name not in {"pandas", "torch"}:
        raise
    LogDataset = None

try:
    from .mode_change import compute_change_scores, detect_change_points, cluster_segments
except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional local ML stack
    if exc.name != "sklearn":
        raise
    compute_change_scores = None
    detect_change_points = None
    cluster_segments = None

__all__ = [
    "LogParser",
    "LogDataset",
    "compute_change_scores",
    "detect_change_points",
    "cluster_segments",
]
