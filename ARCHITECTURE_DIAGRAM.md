# Log Preprocessing Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RAW LOG DATA (CSV)                            │
│  timestamp | event_type | component | severity | message | data         │
│  2026-...  | SENSOR_READ| TempSensor| INFO     | ...     | {'temp':65} │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         LogPreprocessor.fit()                           │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Build Categorical Encoders                                      │   │
│  │  • event_type:  "SENSOR_READ" → 10                             │   │
│  │  • component:   "TempSensor" → 6                               │   │
│  │  • severity:    "INFO" → 0                                     │   │
│  │  • state:       "RUNNING" → 1                                  │   │
│  └────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────┐   │
│  │ Compute Numerical Statistics                                    │   │
│  │  • temperature: mean=52.3, std=19.0                            │   │
│  │  • pressure:    mean=5.5,  std=2.6                             │   │
│  │  • position:    mean=49.6, std=28.7                            │   │
│  └────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      LogPreprocessor.transform()                        │
│                                                                          │
│  For each log entry:                                                    │
│                                                                          │
│  ┌──────────────────────────────┐  ┌──────────────────────────────┐  │
│  │   Categorical → Integer IDs  │  │   Numerical → Normalized     │  │
│  │                               │  │                               │  │
│  │   event_type → ID            │  │   temp → (temp-mean)/std     │  │
│  │   component  → ID            │  │   pressure → normalized      │  │
│  │   severity   → ID            │  │   position → normalized      │  │
│  │   state      → ID            │  │                               │  │
│  └──────────────────────────────┘  └──────────────────────────────┘  │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    FEATURE MATRIX (NumPy Array)                         │
│  Shape: (n_samples, 10)                                                 │
│                                                                          │
│  [[10.0, 6.0, 0.0, 1.0, 3.0, 0.0, -1.41, 0.69, 0.0, 0.0],             │
│   [ 0.0, 3.0, 0.0, 1.0, 2.0, 0.54,  0.0,  0.0,  0.0, 0.0],             │
│   [12.0, 4.0, 1.0, 2.0, 3.0, 0.0,   0.0,  0.0,  0.0, 0.0],             │
│   ...]                                                                   │
│                                                                          │
│  Feature indices:                                                        │
│  [0] event_type_id     [5] position                                    │
│  [1] component_id      [6] pressure                                    │
│  [2] severity_id       [7] temperature                                 │
│  [3] state_id          [8] threshold_exceeded                          │
│  [4] actuator_state    [9] value                                       │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    create_sequences() (Optional)                        │
│  Sliding window with length=10, stride=5                                │
│                                                                          │
│  [[sample_0,  sample_1,  ..., sample_9],     ← Sequence 0             │
│   [sample_5,  sample_6,  ..., sample_14],    ← Sequence 1             │
│   [sample_10, sample_11, ..., sample_19],    ← Sequence 2             │
│   ...]                                                                   │
└──────────────────────────────────────────┬──────────────────────────────┘
                                           │
                                           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     SEQUENCE ARRAY (NumPy Array)                        │
│  Shape: (n_sequences, sequence_length, n_features)                     │
│                                                                          │
│  Used for: Time series models, LSTMs, Transformers                     │
└─────────────────────────────────────────────────────────────────────────┘


USAGE PATHS:
─────────────

Path 1: Scikit-learn Models
  feature_matrix → RandomForestClassifier / SVM / etc.

Path 2: PyTorch Deep Learning
  sequences → torch.FloatTensor → LSTM / Transformer / CNN

Path 3: Clustering / Anomaly Detection
  feature_matrix → KMeans / IsolationForest / etc.

Path 4: Save and Load Later
  preprocessor.save('preprocessor.json')
  preprocessor.load('preprocessor.json')
  new_features = preprocessor.transform(new_logs)


PERSISTENCE:
────────────

Files Generated:
  • preprocessor.json      - Encoder mappings & statistics (~2.5 KB)
  • features.npz           - Processed feature arrays    (~2.3 MB for 10k samples)

Benefits:
  ✓ Consistent encoding across train/test splits
  ✓ Reproducible preprocessing pipeline
  ✓ Easy deployment to production
```

## Data Flow Summary

```
CSV Logs → LogPreprocessor.fit()      → Build encoders & stats
         → LogPreprocessor.transform() → Numerical vectors
         → create_sequences()          → Time series sequences
         → ML Model                    → Predictions
```

## Feature Types

| Type         | Examples                     | Encoding                |
|--------------|------------------------------|-------------------------|
| Categorical  | event_type, component, state | Integer IDs (0, 1, 2..) |
| Numerical    | temperature, pressure        | Z-score normalized      |
| Boolean      | threshold_exceeded           | 0 or 1                  |
| Ordinal      | severity (INFO→CRITICAL)     | Ordered IDs (0→3)       |

## Key Classes & Functions

```python
# Main preprocessor class
LogPreprocessor()
  .fit(logs_df)              # Build encoders
  .transform(logs_df)        # Convert to features
  .fit_transform(logs_df)    # Fit + transform
  .save(filepath)            # Save state
  .load(filepath)            # Load state
  .decode_categorical(...)   # Decode ID back to label
  .get_feature_names()       # List feature names
  .get_feature_info()        # Get statistics

# Sequence creation helper
create_sequences(features, sequence_length, stride)
```
