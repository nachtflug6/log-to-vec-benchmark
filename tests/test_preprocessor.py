"""
Tests for LogPreprocessor
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.log_to_vec.data.preprocessor import LogPreprocessor, create_sequences


@pytest.fixture
def sample_logs():
    """Create sample log data for testing."""
    data = {
        'timestamp': pd.date_range('2026-01-01', periods=100, freq='1s'),
        'event_type': ['SENSOR_READ', 'ACTUATOR_CMD'] * 50,
        'component': ['TemperatureSensor', 'Actuator1'] * 50,
        'severity': ['INFO', 'WARNING'] * 50,
        'message': ['SENSOR_READ in RUNNING state', 'ACTUATOR_CMD in RUNNING state'] * 50,
        'data': [
            "{'temperature': 25.0, 'pressure': 1.5}",
            "{'position': 50.0, 'state': 'OPEN'}"
        ] * 50
    }
    return pd.DataFrame(data)


def test_preprocessor_fit(sample_logs):
    """Test fitting the preprocessor."""
    preprocessor = LogPreprocessor()
    preprocessor.fit(sample_logs)
    
    assert preprocessor.fitted
    assert len(preprocessor.feature_names) > 0
    assert 'event_type' in preprocessor.categorical_encoders
    assert len(preprocessor.categorical_encoders['event_type']) == 2


def test_preprocessor_transform(sample_logs):
    """Test transforming log data."""
    preprocessor = LogPreprocessor()
    feature_matrix = preprocessor.fit_transform(sample_logs)
    
    assert feature_matrix.shape[0] == len(sample_logs)
    assert feature_matrix.shape[1] == len(preprocessor.feature_names)
    assert not np.isnan(feature_matrix).any()


def test_preprocessor_fit_transform_consistency(sample_logs):
    """Test that fit_transform produces same result as fit then transform."""
    preprocessor1 = LogPreprocessor()
    result1 = preprocessor1.fit_transform(sample_logs)
    
    preprocessor2 = LogPreprocessor()
    preprocessor2.fit(sample_logs)
    result2 = preprocessor2.transform(sample_logs)
    
    np.testing.assert_array_equal(result1, result2)


def test_categorical_encoding(sample_logs):
    """Test categorical feature encoding."""
    preprocessor = LogPreprocessor()
    preprocessor.fit(sample_logs)
    
    # Check event type encoding
    assert 'SENSOR_READ' in preprocessor.categorical_encoders['event_type']
    assert 'ACTUATOR_CMD' in preprocessor.categorical_encoders['event_type']
    
    # Check decoding
    event_id = preprocessor.categorical_encoders['event_type']['SENSOR_READ']
    decoded = preprocessor.decode_categorical('event_type', event_id)
    assert decoded == 'SENSOR_READ'


def test_normalization(sample_logs):
    """Test numerical feature normalization."""
    preprocessor = LogPreprocessor()
    
    # With normalization
    X_normalized = preprocessor.fit_transform(sample_logs, normalize=True)
    
    # Without normalization
    X_raw = preprocessor.transform(sample_logs, normalize=False)
    
    # Shapes should be the same
    assert X_normalized.shape == X_raw.shape
    
    # Values should be different (for numerical features)
    assert not np.array_equal(X_normalized, X_raw)


def test_create_sequences():
    """Test sequence creation."""
    feature_matrix = np.random.randn(100, 10)
    
    sequences = create_sequences(feature_matrix, sequence_length=10, stride=5)
    
    assert sequences.ndim == 3
    assert sequences.shape[1] == 10  # sequence_length
    assert sequences.shape[2] == 10  # n_features
    
    # Check correct number of sequences
    expected_n_seq = (100 - 10) // 5 + 1
    assert sequences.shape[0] == expected_n_seq


def test_save_load_preprocessor(sample_logs, tmp_path):
    """Test saving and loading preprocessor state."""
    # Fit preprocessor
    preprocessor1 = LogPreprocessor()
    preprocessor1.fit(sample_logs)
    
    # Save
    save_path = tmp_path / "preprocessor.json"
    preprocessor1.save(str(save_path))
    
    # Load
    preprocessor2 = LogPreprocessor()
    preprocessor2.load(str(save_path))
    
    # Check state is preserved
    assert preprocessor2.fitted
    assert preprocessor2.feature_names == preprocessor1.feature_names
    assert preprocessor2.categorical_encoders == preprocessor1.categorical_encoders
    
    # Transform should produce same result
    X1 = preprocessor1.transform(sample_logs)
    X2 = preprocessor2.transform(sample_logs)
    np.testing.assert_array_equal(X1, X2)


def test_feature_info(sample_logs):
    """Test feature information retrieval."""
    preprocessor = LogPreprocessor()
    preprocessor.fit(sample_logs)
    
    info = preprocessor.get_feature_info()
    
    assert 'n_features' in info
    assert 'feature_names' in info
    assert 'categorical_features' in info
    assert 'numerical_features' in info
    
    assert info['n_features'] == len(preprocessor.feature_names)


def test_unknown_categorical_values(sample_logs):
    """Test handling of unknown categorical values."""
    preprocessor = LogPreprocessor()
    preprocessor.fit(sample_logs)
    
    # Create test data with unknown event type
    test_data = sample_logs.copy()
    test_data.loc[0, 'event_type'] = 'UNKNOWN_EVENT'
    
    # Should not raise error
    X = preprocessor.transform(test_data)
    
    # Unknown value should be encoded as -1
    assert X[0, 0] == -1  # Assuming event_type_id is first feature


def test_missing_numerical_values():
    """Test handling of missing numerical values."""
    data = {
        'timestamp': pd.date_range('2026-01-01', periods=10, freq='1s'),
        'event_type': ['SENSOR_READ'] * 10,
        'component': ['Sensor1'] * 10,
        'severity': ['INFO'] * 10,
        'message': ['SENSOR_READ in RUNNING state'] * 10,
        'data': [''] * 10  # Empty data
    }
    logs_df = pd.DataFrame(data)
    
    preprocessor = LogPreprocessor()
    X = preprocessor.fit_transform(logs_df)
    
    # Should not have NaN values
    assert not np.isnan(X).any()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
