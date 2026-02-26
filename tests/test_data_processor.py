"""Tests for data processor service."""
import pytest
import pandas as pd
import numpy as np
from app.services.data_processor import DataProcessor


def test_preprocess_data(data_processor, sample_price_data):
    """Test data preprocessing."""
    # Add some NaN values
    sample_price_data.loc[5, 'close'] = np.nan
    
    processed = data_processor.preprocess_data(sample_price_data)
    
    assert len(processed) > 0
    assert processed['close'].isna().sum() == 0
    assert 'timestamp' in processed.columns


def test_handle_missing_values(data_processor, sample_price_data):
    """Test missing value handling."""
    # Add NaN values
    sample_price_data.loc[10:15, 'close'] = np.nan
    
    processed = data_processor._handle_missing_values(sample_price_data)
    
    assert processed['close'].isna().sum() == 0

