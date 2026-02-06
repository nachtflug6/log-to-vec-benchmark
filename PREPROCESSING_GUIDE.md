# Log Preprocessing Guide

This guide explains how to preprocess log data into numerical feature vectors for machine learning.

## Overview

The `LogPreprocessor` transforms raw log data into numerical representations suitable for ML models. It handles:

- **Categorical Features**: Event types, components, severity levels, states → Integer IDs (0, 1, 2, ...)
- **Numerical Features**: Temperatures, pressures, positions → Normalized continuous values
- **Boolean Features**: Flags and states → Binary values (0, 1)

## Feature Representation

Each log entry is converted to a feature vector. For example:

```
Log Entry:
{
  "event_type": "SENSOR_READ",
  "component": "TemperatureSensor",
  "severity": "INFO",
  "message": "SENSOR_READ in RUNNING state",
  "data": "{'temperature': 65.5, 'pressure': 3.2}"
}

↓ Preprocessor ↓

Feature Vector:
[10, 0, 0, 1, 3, 0.0, -1.41, 0.69, 0.0, 0.0]
 │   │  │  │  │   │     │     │     │     │
 │   │  │  │  │   │     │     │     │     └─ value (normalized)
 │   │  │  │  │   │     │     │     └─────── threshold_exceeded (0/1)
 │   │  │  │  │   │     │     └─────────── temperature (normalized)
 │   │  │  │  │   │     └─────────────── pressure (normalized)
 │   │  │  │  │   └───────────────────── position (normalized)
 │   │  │  │  └──────────────────────── actuator_state (0-3)
 │   │  │  └─────────────────────────── state_id (0=MAINTENANCE, 1=RUNNING, 2=STOPPED)
 │   │  └────────────────────────────── severity_id (0=INFO, 1=WARNING, 2=ERROR, 3=CRITICAL)
 │   └───────────────────────────────── component_id (0-6)
 └───────────────────────────────────── event_type_id (0-13)
```

## Quick Start

### 1. Preprocess Logs

```bash
# Basic usage
python examples/preprocess_logs.py --input data/toy_logs.csv

# With sequences (sliding window)
python examples/preprocess_logs.py \
    --input data/toy_logs.csv \
    --sequence-length 10 \
    --stride 5 \
    --output data/processed_features.npz
```

### 2. Use in Python

```python
from log_to_vec.data import LogPreprocessor
import pandas as pd

# Load logs
logs_df = pd.read_csv("data/toy_logs.csv")

# Fit and transform
preprocessor = LogPreprocessor()
feature_matrix = preprocessor.fit_transform(logs_df)

# feature_matrix shape: (n_samples, n_features)
print(f"Shape: {feature_matrix.shape}")
print(f"Features: {preprocessor.get_feature_names()}")

# Save preprocessor for later use
preprocessor.save("preprocessor.json")
```

### 3. Create Sequences

```python
from log_to_vec.data import create_sequences

# Create overlapping sequences for time series models
sequences = create_sequences(
    feature_matrix,
    sequence_length=10,  # 10 log entries per sequence
    stride=5             # 50% overlap
)

# sequences shape: (n_sequences, sequence_length, n_features)
print(f"Sequences: {sequences.shape}")
```

## Features Extracted

### Categorical Features (Encoded as Integer IDs)

| Feature | Description | Values |
|---------|-------------|--------|
| `event_type_id` | Type of log event | 0-13 (14 unique event types) |
| `component_id` | System component | 0-6 (7 unique components) |
| `severity_id` | Log severity level | 0=INFO, 1=WARNING, 2=ERROR, 3=CRITICAL |
| `state_id` | System state | 0=MAINTENANCE, 1=RUNNING, 2=STOPPED |

### Numerical Features (Normalized)

| Feature | Description | Original Range | Normalized |
|---------|-------------|----------------|------------|
| `temperature` | Temperature reading | 20-85°C | z-score normalized |
| `pressure` | Pressure reading | 1-10 bar | z-score normalized |
| `position` | Actuator position | 0-100% | z-score normalized |
| `value` | Alarm threshold value | 80-100 | z-score normalized |
| `actuator_state` | Actuator state | OPEN=0, CLOSED=1, MOVING=2, UNKNOWN=3 | Not normalized |
| `threshold_exceeded` | Alarm flag | False=0, True=1 | Binary |

## Advanced Usage

### Load Preprocessor and Transform New Data

```python
# Load pre-fitted preprocessor
preprocessor = LogPreprocessor()
preprocessor.load("preprocessor.json")

# Transform new log data
new_logs_df = pd.read_csv("new_logs.csv")
new_features = preprocessor.transform(new_logs_df)
```

### Access Feature Information

```python
# Get feature names
feature_names = preprocessor.get_feature_names()

# Get detailed info
info = preprocessor.get_feature_info()
print(f"Total features: {info['n_features']}")
print(f"Categorical: {info['categorical_features']}")
print(f"Numerical: {info['numerical_features']}")
```

### Decode Categorical Values

```python
# Decode a categorical feature
event_id = 10
event_name = preprocessor.decode_categorical('event_type', event_id)
print(f"Event ID {event_id} = {event_name}")
```

## Visualization

Visualize the preprocessed features:

```bash
python examples/visualize_features.py \
    --input data/processed_features.npz \
    --output-dir data/
```

This generates:
- `feature_distributions.png` - Histograms of all features
- `feature_correlations.png` - Correlation matrix heatmap
- `time_series_features.png` - Time series plots
- `sequence_visualization.png` - Sequence heatmaps

## Normalization

By default, numerical features are normalized using z-score normalization:

```
normalized_value = (value - mean) / std
```

This ensures all features are on a similar scale, which is important for many ML algorithms.

To disable normalization:

```python
feature_matrix = preprocessor.transform(logs_df, normalize=False)
```

## Missing Values

Missing or NaN values in numerical features are replaced with 0 (after normalization).

Unknown categorical values (not seen during fitting) are encoded as -1.

## Example Output

```
Fitting preprocessor on training data...
  Event types: 14 unique values
  Components: 7 unique values
  Severity levels: 4 unique values
  States: 3 unique values
  temperature: mean=52.311, std=18.973
  pressure: mean=5.513, std=2.611
  position: mean=49.566, std=28.658
  value: mean=89.849, std=5.722
Preprocessor fitted. Total features: 10

Feature matrix shape: (10000, 10)
```

## Integration with Models

The preprocessed features can be used directly with ML models:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Prepare data
X = feature_matrix
y = labels  # Your target labels

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y)
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

Or with PyTorch for deep learning:

```python
import torch
from torch.utils.data import TensorDataset, DataLoader

# Create PyTorch dataset
tensor_X = torch.FloatTensor(sequences)
tensor_y = torch.LongTensor(labels)
dataset = TensorDataset(tensor_X, tensor_y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## See Also

- [toy_log_generator.py](../examples/toy_log_generator.py) - Generate synthetic log data
- [preprocess_logs.py](../examples/preprocess_logs.py) - Preprocessing script
- [preprocessor.py](../src/log_to_vec/data/preprocessor.py) - Preprocessor implementation
