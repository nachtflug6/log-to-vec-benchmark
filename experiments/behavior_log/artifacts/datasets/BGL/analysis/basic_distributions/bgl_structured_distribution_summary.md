# BGL Structured Basic Distribution Analysis

- Input file: `/home/lyra/projects/log-to-vec-benchmark/experiments/behavior_log/artifacts/datasets/BGL/structured/BGL_structured.csv`
- Total rows: 4,631,261
- Max rows: all

## Label
- Unique values: 32
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 92.83% |
| top_5 | 99.08% |
| top_10 | 99.89% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## BinaryLabel
- Unique values: 2
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 92.83% |
| top_5 | 100.00% |
| top_10 | 100.00% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## EventId
- Unique values: 320
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 36.85% |
| top_5 | 62.93% |
| top_10 | 76.17% |
| top_20 | 85.66% |
| top_50 | 94.48% |
| top_100 | 99.18% |

## Node
- Unique values: 69,227
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 3.29% |
| top_5 | 7.89% |
| top_10 | 9.08% |
| top_20 | 9.81% |
| top_50 | 10.66% |
| top_100 | 11.67% |

## NodeRepeat
- Unique values: 69,233
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 3.29% |
| top_5 | 7.89% |
| top_10 | 9.08% |
| top_20 | 9.81% |
| top_50 | 10.66% |
| top_100 | 11.67% |

## Type
- Unique values: 7
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 98.18% |
| top_5 | 100.00% |
| top_10 | 100.00% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## Component
- Unique values: 14
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 91.69% |
| top_5 | 99.93% |
| top_10 | 100.00% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## Level
- Unique values: 10
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 79.88% |
| top_5 | 99.96% |
| top_10 | 100.00% |
| top_20 | 100.00% |
| top_50 | 100.00% |
| top_100 | 100.00% |

## Date
- Unique values: 215
- Non-empty rows: 4,631,261
- Missing rows: 0
- Coverage of rows: 100.00%

| Top-K | Cumulative share of non-empty rows |
|---|---:|
| top_1 | 8.24% |
| top_5 | 30.26% |
| top_10 | 46.33% |
| top_20 | 64.37% |
| top_50 | 84.94% |
| top_100 | 93.75% |

## Node Event Traces
- Output file: `/home/lyra/projects/log-to-vec-benchmark/experiments/behavior_log/artifacts/datasets/BGL/analysis/basic_distributions/Node_event_traces.csv`
- Nodes: 69,227
- Features column: event sequence for each node-level trace
- Average trace length: 66.90
- Max trace length: 152,320
- Average latency: 11760944.99 seconds
- Max latency: 18,541,197 seconds

| Label | Nodes |
|---|---:|
| Anomaly | 31,361 |
| Normal | 37,866 |

## Node Occurrence Matrix
- Output file: `/home/lyra/projects/log-to-vec-benchmark/experiments/behavior_log/artifacts/datasets/BGL/analysis/basic_distributions/Node_event_occurrence_matrix.csv`
- Nodes: 69,227
- Columns: Node, Label, and one count column per EventId
- Event feature columns: 320
- Average non-zero event types per node: 10.22
- Max non-zero event types per node: 142
