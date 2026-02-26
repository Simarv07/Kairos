import logging
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.preprocessing import StandardScaler

from config import settings

logger = logging.getLogger(__name__)


# Service for preprocessing and processing price data
class DataProcessor:
    def __init__(self):
        self.scaler = None
    
    def preprocess_data(
        self,
        df: pd.DataFrame,
        normalize: bool = False
    ) -> pd.DataFrame:
        # Preprocesses price data: handles timestamps, missing values, optional normalization
        df = df.copy()
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Normalize if requested
        if normalize:
            df = self._normalize_data(df)
        
        return df
    
    # Handles missing values in the dataframe using forward/backward fill
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        # Forward fill for price data
        price_columns = ['open', 'high', 'low', 'close']
        df[price_columns] = df[price_columns].ffill()
        # Backward fill so the last row (current date) is never dropped when it has partial data
        df[price_columns] = df[price_columns].bfill()
        
        # For volume, fill with 0
        if 'volume' in df.columns:
            df['volume'] = df['volume'].fillna(0)
        
        # Drop only rows that still have NaN (e.g. leading rows before first valid); keep last row for current date
        df = df.dropna(subset=price_columns + (['volume'] if 'volume' in df.columns else []))
        
        return df
    
    # Normalizes price data using StandardScaler
    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        price_columns = ['open', 'high', 'low', 'close']
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            df[price_columns] = self.scaler.fit_transform(df[price_columns])
        else:
            df[price_columns] = self.scaler.transform(df[price_columns])
        
        return df
    
    def split_train_test(
        self,
        df: pd.DataFrame,
        split_ratio: float = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Splits data into train and test sets based on split ratio
        if split_ratio is None:
            split_ratio = settings.TRAIN_TEST_SPLIT
        
        split_idx = int(len(df) * split_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        logger.info(f"Split data: {len(train_df)} train, {len(test_df)} test")
        return train_df, test_df

