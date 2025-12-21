"""
Tests for data cleaning utilities
"""
import pytest
import pandas as pd
import numpy as np
from src.task1.data_cleaner import DataCleaner

def test_fraud_data_cleaning():
    """Test comprehensive fraud dataset cleaning"""
    cleaner = DataCleaner(dataset_type='fraud')
    
    # Create test data with known issues
    test_data = pd.DataFrame({
        'purchase_value': [100, 200, np.nan, 300],
        'age': [25, 30, np.nan, 35],
        'browser': ['Chrome', 'Firefox', np.nan, 'Chrome'],
        'class': [0, 1, 0, 1]
    })
    
    # Test cleaning
    cleaned = cleaner.clean_fraud_dataset(test_data)
    
    # Assert no missing values
    assert cleaned.isnull().sum().sum() == 0
    
    # Assert data types
    assert pd.api.types.is_numeric_dtype(cleaned['purchase_value'])
    assert pd.api.types.is_numeric_dtype(cleaned['age'])
    
    print("✓ Fraud data cleaning tests passed")

def test_credit_data_cleaning():
    """Test comprehensive credit dataset cleaning"""
    cleaner = DataCleaner(dataset_type='credit')
    
    # Create test data
    test_data = pd.DataFrame({
        'Time': [1, 2, 3, 4],
        'Amount': [100.0, 200.0, np.nan, 400.0],
        'V1': [0.1, 0.2, 0.3, 0.4],
        'Class': [0, 1, 0, 1]
    })
    
    # Test cleaning
    cleaned = cleaner.clean_credit_dataset(test_data)
    
    # Assert no missing values
    assert cleaned.isnull().sum().sum() == 0
    
    # Assert data types
    assert pd.api.types.is_numeric_dtype(cleaned['Amount'])
    assert pd.api.types.is_numeric_dtype(cleaned['V1'])
    
    print("✓ Credit data cleaning tests passed")