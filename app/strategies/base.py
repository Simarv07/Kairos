from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd


# Base class for all trading strategies
class BaseStrategy(ABC):
    def __init__(self, **kwargs):
        self.parameters = kwargs
        self.is_fitted = False
        self.current_price: Optional[float] = None
    
    # Initializes or trains the strategy on historical data
    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        if train_data.empty:
            raise ValueError("Training data is empty")
        
        # Default implementation: just mark as fitted and store last price
        if 'close' in train_data.columns:
            self.current_price = float(train_data['close'].iloc[-1])
        
        self.is_fitted = True
        return {'is_fitted': True}
    
    # Generates predictions for future prices (returns current price by default)
    def predict(self, steps: int = 1) -> List[float]:
        # Default implementation: return current price if available
        if self.current_price is not None:
            return [self.current_price] * steps
        return [0.0] * steps
    
    # Generates trading signals (BUY, SELL, or HOLD) - must be implemented by subclasses
    @abstractmethod
    def generate_signals(self, current_price: float, predicted_price: float, threshold: float = 0.01) -> str:
        pass
    
    # Returns strategy parameters
    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters

