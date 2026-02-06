# Log Preprocessing Quick Reference

## Installation
```bash
pip install -e .
```

## Basic Usage (3 Lines)
```python
from log_to_vec.data import LogPreprocessor
import pandas as pd

logs_df = pd.read_csv("logs.csv")
preprocessor = LogPreprocessor()
features = preprocessor.fit_transform(logs_df)  # Done! Shape: (n_samples, 10)
```

## Common Tasks

### 1. Preprocess from Command Line
```bash
python examples/preprocess_logs.py --input data/toy_logs.csv
```

### 2. Create Sequences
```python
from log_to_vec.data import create_sequences
sequences = create_sequences(features, sequence_length=10, stride=5)
```

### 3. Save & Load Preprocessor
```python
# Save
preprocessor.save("preprocessor.json")

# Load
preprocessor = LogPreprocessor()
preprocessor.load("preprocessor.json")
new_features = preprocessor.transform(new_logs_df)
```

### 4. Train a Classifier
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(features, labels)
predictions = model.predict(test_features)
```

### 5. Use with PyTorch
```python
import torch
from torch.utils.data import TensorDataset, DataLoader

X = torch.FloatTensor(sequences)
y = torch.LongTensor(labels)
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32)
```

### 6. Decode Features
```python
event_id = int(features[0, 0])
event_name = preprocessor.decode_categorical('event_type', event_id)
```

### 7. Get Feature Info
```python
names = preprocessor.get_feature_names()
info = preprocessor.get_feature_info()
```

## Output Format

**Feature Vector** (10 features per sample):
```
[event_type_id, component_id, severity_id, state_id, 
 actuator_state, position, pressure, temperature, 
 threshold_exceeded, value]
```

**Example:**
```
[10.0, 6.0, 0.0, 1.0, 3.0, 0.0, -1.41, 0.69, 0.0, 0.0]
  │     │    │    │    │    │     │      │     │    └─ value
  │     │    │    │    │    │     │      │     └────── threshold_exceeded
  │     │    │    │    │    │     │      └──────────── temperature (normalized)
  │     │    │    │    │    │     └─────────────────── pressure (normalized)
  │     │    │    │    │    └───────────────────────── position (normalized)
  │     │    │    │    └────────────────────────────── actuator_state
  │     │    │    └─────────────────────────────────── state_id
  │     │    └──────────────────────────────────────── severity_id
  │     └───────────────────────────────────────────── component_id
  └─────────────────────────────────────────────────── event_type_id
```

## Example Scripts

| Script | Purpose |
|--------|---------|
| `examples/preprocess_logs.py` | Main preprocessing script |
| `examples/visualize_features.py` | Visualize feature distributions |
| `examples/train_classifier.py` | Train ML classifier |
| `examples/load_features.py` | Load and inspect features |
| `examples/complete_workflow.py` | Full pipeline demo |
| `examples/demo_pipeline.sh` | Automated demo script |

## Feature Details

### Categorical Features (Integer IDs)
- `event_type_id`: 0-13 (14 event types)
- `component_id`: 0-6 (7 components)  
- `severity_id`: 0=INFO, 1=WARNING, 2=ERROR, 3=CRITICAL
- `state_id`: 0=MAINTENANCE, 1=RUNNING, 2=STOPPED
- `actuator_state`: 0=OPEN, 1=CLOSED, 2=MOVING, 3=UNKNOWN

### Numerical Features (Z-score Normalized)
- `temperature`: Sensor temperature reading
- `pressure`: Sensor pressure reading
- `position`: Actuator position (0-100%)
- `value`: Alarm threshold value
- `threshold_exceeded`: Binary flag (0 or 1)

## Testing
```bash
pytest tests/test_preprocessor.py -v
```

## Documentation
- `PREPROCESSING_GUIDE.md` - Comprehensive guide
- `ARCHITECTURE_DIAGRAM.md` - Visual architecture
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

## Troubleshooting

**Q: Unknown categorical values?**  
A: Encoded as -1. Train on representative data.

**Q: NaN in features?**  
A: Replaced with 0 after normalization.

**Q: Change feature order?**  
A: Use `get_feature_names()` to see order, then index accordingly.

**Q: Disable normalization?**  
A: `preprocessor.transform(logs_df, normalize=False)`

## Tips

✓ Always fit on training data, then transform test data  
✓ Save preprocessor state for consistent encoding  
✓ Check `get_feature_info()` to understand your data  
✓ Use sequences for time-series models  
✓ Use feature matrix for traditional ML  
