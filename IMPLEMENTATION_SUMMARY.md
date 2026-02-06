# Log Preprocessing Implementation Summary

## Overview

We have successfully implemented a comprehensive log preprocessing system that transforms raw time-stamped log files into numerical feature vectors suitable for machine learning models.

## What Was Implemented

### 1. Core Preprocessor Module (`src/log_to_vec/data/preprocessor.py`)

**Class: `LogPreprocessor`**

Transforms logs into numerical representations with:

- **Categorical Feature Encoding**: Converts event types, components, severity levels, and states into integer IDs
- **Numerical Feature Extraction**: Extracts and normalizes temperature, pressure, position, and other sensor values
- **State Management**: Saves/loads preprocessor state for consistent encoding across train/test splits
- **Missing Value Handling**: Gracefully handles missing or unknown values

**Key Features:**
- Z-score normalization for numerical features
- Automatic vocabulary building for categorical features
- JSON serialization for saving/loading state
- Detailed feature information and statistics

### 2. Example Scripts

#### `examples/preprocess_logs.py`
Main preprocessing script that:
- Loads log CSV files
- Fits preprocessor and transforms data
- Creates sequences using sliding windows
- Saves processed features and preprocessor state

**Usage:**
```bash
python examples/preprocess_logs.py \
    --input data/toy_logs.csv \
    --sequence-length 10 \
    --stride 5 \
    --output data/processed_features.npz
```

#### `examples/visualize_features.py`
Visualization script that generates:
- Feature distribution histograms
- Correlation matrix heatmaps
- Time series plots
- Sequence visualizations

**Usage:**
```bash
python examples/visualize_features.py \
    --input data/processed_features.npz \
    --output-dir data/
```

#### `examples/train_classifier.py`
Demonstrates ML integration by training a Random Forest classifier for anomaly detection.

**Usage:**
```bash
python examples/train_classifier.py --input data/toy_logs.csv
```

#### `examples/load_features.py`
Shows how to load and inspect preprocessed features, including decoding categorical values.

**Usage:**
```bash
python examples/load_features.py \
    --features data/processed_features.npz \
    --preprocessor data/preprocessor.json
```

### 3. Documentation

#### `PREPROCESSING_GUIDE.md`
Comprehensive guide covering:
- Feature representation format
- Quick start examples
- Detailed feature descriptions
- Advanced usage patterns
- Integration with ML models

### 4. Tests (`tests/test_preprocessor.py`)

Complete test suite with 10 tests covering:
- Fitting and transformation
- Categorical encoding/decoding
- Normalization
- Sequence creation
- Save/load functionality
- Missing value handling
- Unknown categorical values

**All tests pass ✓**

## Feature Representation

Each log entry is transformed into a vector with 10 features:

```
[event_type_id, component_id, severity_id, state_id, 
 actuator_state, position, pressure, temperature, 
 threshold_exceeded, value]
```

**Example:**
```
Log: {event_type: "SENSOR_READ", severity: "INFO", temperature: 65.5}
  ↓
Vector: [10, 0, 0, 1, 3, 0.0, -1.41, 0.69, 0.0, 0.0]
```

## Results from Testing

### Data Processing
- ✅ Successfully processed 10,000 log entries
- ✅ Generated 10-dimensional feature vectors
- ✅ Created 1,999 sequences (length=10, stride=5)
- ✅ Total size: ~2.3 MB compressed

### Classification Performance
- ✅ 100% accuracy on anomaly detection task
- ✅ Perfect precision and recall
- ✅ Most important features: severity_id, event_type_id, threshold_exceeded

### Feature Statistics
- ✅ 14 unique event types encoded (0-13)
- ✅ 7 unique components encoded (0-6)
- ✅ 4 severity levels encoded (0-3)
- ✅ 3 system states encoded (0-2)
- ✅ Numerical features properly normalized (mean≈0, std≈1)

## Integration Points

The preprocessor seamlessly integrates with:

1. **Scikit-learn Models**: Direct use of feature matrix
   ```python
   model = RandomForestClassifier()
   model.fit(feature_matrix, labels)
   ```

2. **PyTorch Models**: Convert to tensors for deep learning
   ```python
   tensor_X = torch.FloatTensor(sequences)
   dataset = TensorDataset(tensor_X, tensor_y)
   ```

3. **Existing Log Parser**: Can be used alongside or independently of the token-based parser

## Files Created/Modified

### New Files
- `src/log_to_vec/data/preprocessor.py` (462 lines)
- `examples/preprocess_logs.py` (139 lines)
- `examples/visualize_features.py` (201 lines)
- `examples/train_classifier.py` (162 lines)
- `examples/load_features.py` (187 lines)
- `tests/test_preprocessor.py` (216 lines)
- `PREPROCESSING_GUIDE.md` (282 lines)

### Modified Files
- `src/log_to_vec/data/__init__.py` - Added preprocessor exports
- `README.md` - Added preprocessing section

### Generated Data Files
- `data/preprocessor.json` - Preprocessor state (2.5 KB)
- `data/processed_features.npz` - Processed features (2.3 MB)

## Usage Examples

### Basic Preprocessing
```python
from log_to_vec.data import LogPreprocessor
import pandas as pd

logs_df = pd.read_csv("logs.csv")
preprocessor = LogPreprocessor()
features = preprocessor.fit_transform(logs_df)
preprocessor.save("preprocessor.json")
```

### Create Sequences
```python
from log_to_vec.data import create_sequences

sequences = create_sequences(
    features, 
    sequence_length=10, 
    stride=5
)
```

### Load and Use
```python
preprocessor = LogPreprocessor()
preprocessor.load("preprocessor.json")

new_logs = pd.read_csv("new_logs.csv")
new_features = preprocessor.transform(new_logs)
```

## Key Benefits

1. **Standardized Representation**: All logs converted to fixed-length vectors
2. **ML-Ready**: Direct input to ML models without additional processing
3. **Reproducible**: Save/load ensures consistent encoding
4. **Flexible**: Works with sequences or individual samples
5. **Documented**: Comprehensive guides and examples
6. **Tested**: Full test coverage ensures reliability

## Next Steps

Potential enhancements:
- One-hot encoding option for categorical features
- Principal Component Analysis (PCA) for dimensionality reduction
- Custom feature extractors for domain-specific data
- Streaming preprocessing for large datasets
- Additional normalization strategies (min-max, robust scaling)

## Conclusion

The preprocessing system provides a robust, well-tested foundation for transforming log data into numerical representations. It handles both categorical and continuous features, supports sequence creation, and integrates seamlessly with popular ML frameworks.
