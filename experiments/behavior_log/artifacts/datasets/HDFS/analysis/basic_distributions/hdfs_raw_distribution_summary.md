# HDFS Raw Basic Distribution Analysis

- Input file: `/home/lyra/projects/log-to-vec-benchmark/experiments/behavior_log/artifacts/datasets/HDFS/raw/HDFS_raw.csv`
- Total rows: 11,167,740
- Max rows: all

## EventId
- Unique values: 46
- Non-empty rows: 11,167,740
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 15.43% |
| top_5 | 73.95% |
| top_10 | 99.70% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## BlockId
- Unique values: 575,061
- Non-empty rows: 11,167,740
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 0.00% |
| top_5 | 0.01% |
| top_10 | 0.02% |
| top_20 | 0.05% |
| top_50 | 0.10% |
| top_100 | 0.12% |

## Label
- Unique values: 2
- Non-empty rows: 11,167,740
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 97.47% |
| top_5 | 100.00% |
| top_10 | 100.00% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## Level
- Unique values: 2
- Non-empty rows: 11,167,740
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 96.76% |
| top_5 | 100.00% |
| top_10 | 100.00% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## Pid
- Unique values: 27,799
- Non-empty rows: 11,167,740
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 11.48% |
| top_5 | 28.16% |
| top_10 | 42.43% |
| top_20 | 46.82% |
| top_50 | 46.90% |
| top_100 | 47.04% |

## Component
- Unique values: 9
- Non-empty rows: 11,167,740
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 33.12% |
| top_5 | 99.92% |
| top_10 | 100.00% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## Event Traces
- Output file: `/home/lyra/projects/log-to-vec-benchmark/experiments/behavior_log/artifacts/datasets/HDFS/analysis/basic_distributions/Event_traces.csv`
- Blocks: 575,061
- Features column: event sequence for each block-level trace
- Average trace length: 19.42
- Max trace length: 297
- Average latency: 16789.47 seconds
- Max latency: 54,025 seconds

| Label | Blocks |
|---|---:|
| Anomaly | 16,838 |
| Normal | 558,223 |

## Event Occurrence Matrix
- Output file: `/home/lyra/projects/log-to-vec-benchmark/experiments/behavior_log/artifacts/datasets/HDFS/analysis/basic_distributions/Event_occurrence_matrix.csv`
- Blocks: 575,061
- Columns: BlockId, Label, and one count column per EventId
- Event feature columns: 46
- Average non-zero event types per block: 7.24
- Max non-zero event types per block: 26
