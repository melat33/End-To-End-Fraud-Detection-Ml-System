"""
Tests for data transformation pipeline
"""
import pytest
import pandas as pd
import numpy as np
from src.task1.data_transformer import DataTransformer

def test_transformation_pipeline():
    """Test complete transformation pipeline"""
    
    # Create test data with mixed types
    X_train = pd.DataFrame({
        'numeric_feature': [1, 2, 3, 4, 5],
        'categorical_feature': ['A', 'B', 'A', 'C', 'B']
    })
    
    X_test = pd.DataFrame({
        'numeric_feature': [6, 7, 8],
        'categorical_feature': ['A', 'D', 'B']  # D is unseen in training
    })
    
    # Create transformer
    transformer = DataTransformer({
        'numeric_scaler': 'standard',
        'categorical_encoder': 'onehot'
    })
    
    # Fit on training
    X_train_transformed = transformer.fit_transform(X_train)
    
    # Transform test (with unseen category)
    X_test_transformed = transformer.transform(X_test)
    
    # Assertions
    assert X_train_transformed.shape[0] == 5
    assert X_test_transformed.shape[0] == 3
    assert transformer.fitted == True
    
    print("✓ Transformation pipeline tests passed")