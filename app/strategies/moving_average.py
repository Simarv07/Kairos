import logging
from collections import deque
from typing import Dict, Any, List, Optional

import pandas as pd

from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


# Moving Average crossover strategy that generates BUY/SELL signals based on MA crossovers
class MovingAverageStrategy(BaseStrategy):
    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 50,
        **kwargs
    ):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("Moving average windows must be positive integers")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        
        super().__init__(
            short_window=short_window,
            long_window=long_window,
            **kwargs
        )
        
        self.short_window = short_window
        self.long_window = long_window
        self.price_history: deque = deque(maxlen=long_window)
        self.current_price: Optional[float] = None
        self.training_metrics: Dict[str, Any] = {}
        self.is_fitted = False
    
    # Initializes the moving average strategy with training data (signal-based, no model training)
    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        if train_data.empty or 'close' not in train_data.columns:
            raise ValueError("Training data must include 'close' prices")
        
        closes = train_data['close'].dropna().tolist()
        if len(closes) < self.long_window:
            raise ValueError(
                f"Insufficient data for moving averages (need at least {self.long_window} closes)"
            )
        
        for price in closes[-self.long_window:]:
            self.price_history.append(price)
        self.current_price = self.price_history[-1]
        
        short_ma = self._calculate_short_ma()
        long_ma = self._calculate_long_ma()
        
        self.training_metrics = {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "initial_short_ma": short_ma,
            "initial_long_ma": long_ma
        }
        self.is_fitted = True
        
        logger.info(
            "MovingAverageStrategy initialized with short_ma=%.4f, long_ma=%.4f",
            short_ma,
            long_ma
        )
        
        return self.training_metrics
    
    # Updates internal buffers with the latest market data
    def update_market_data(self, row: pd.Series):
        price = float(row.get('close', 0.0))
        if price <= 0:
            return
        
        self.current_price = price
        self.price_history.append(price)
    
    # Returns current price for compatibility (signal-based strategy, doesn't predict prices)
    def predict(self, steps: int = 1) -> List[float]:
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before prediction")
        
        price = self.current_price if self.current_price is not None else 0.0
        return [price for _ in range(steps)]
    
    def generate_signals(
        self,
        current_price: float,
        predicted_price: float,
        threshold: float = 0.0
    ) -> str:
        # Generates BUY/SELL/HOLD signals based on moving average crossover with optional threshold buffer
        if len(self.price_history) < self.long_window:
            return 'HOLD'
        
        short_ma = self._calculate_short_ma()
        long_ma = self._calculate_long_ma()
        
        if long_ma == 0:
            return 'HOLD'
        
        diff_pct = (short_ma - long_ma) / long_ma
        if diff_pct > threshold:
            return 'BUY'
        if diff_pct < -threshold:
            return 'SELL'
        return 'HOLD'
    
    def get_parameters(self) -> Dict[str, Any]:
        params = super().get_parameters()
        params.update({
            "short_window": self.short_window,
            "long_window": self.long_window
        })
        return params
    
    def _calculate_short_ma(self) -> float:
        data = list(self.price_history)[-self.short_window:]
        return float(sum(data) / len(data)) if data else 0.0
    
    def _calculate_long_ma(self) -> float:
        data = list(self.price_history)
        return float(sum(data) / len(data)) if data else 0.0

