import logging
from collections import deque
from typing import Dict, Any, List, Optional

import pandas as pd

from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def _ema_value(prices: List[float], period: int) -> float:
    """Compute EMA over the last `period` prices. EMA = alpha * price + (1-alpha) * prev_EMA, alpha = 2/(period+1)."""
    if not prices or len(prices) < period:
        return 0.0
    alpha = 2.0 / (period + 1)
    ema = sum(prices[:period]) / period  # seed with SMA of first `period` values
    for p in prices[period:]:
        ema = alpha * p + (1 - alpha) * ema
    return float(ema)


# EMA crossover strategy: BUY when short EMA > long EMA, SELL when short EMA < long EMA
class EMAStrategy(BaseStrategy):
    def __init__(
        self,
        short_window: int = 12,
        long_window: int = 26,
        **kwargs
    ):
        if short_window <= 0 or long_window <= 0:
            raise ValueError("EMA windows must be positive integers")
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")

        super().__init__(
            short_window=short_window,
            long_window=long_window,
            **kwargs
        )
        self.short_window = short_window
        self.long_window = long_window
        self.price_history: deque = deque(maxlen=long_window + 50)
        self.current_price: Optional[float] = None
        self.training_metrics: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        if train_data.empty or "close" not in train_data.columns:
            raise ValueError("Training data must include 'close' prices")

        closes = train_data["close"].dropna().tolist()
        if len(closes) < self.long_window:
            raise ValueError(
                f"Insufficient data for EMA (need at least {self.long_window} closes)"
            )

        for price in closes[-self.long_window - 50 :]:
            self.price_history.append(price)
        self.current_price = self.price_history[-1]

        short_ema = self._calculate_short_ema()
        long_ema = self._calculate_long_ema()
        self.training_metrics = {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "initial_short_ema": short_ema,
            "initial_long_ema": long_ema,
        }
        self.is_fitted = True
        logger.info(
            "EMAStrategy initialized with short_ema=%.4f, long_ema=%.4f",
            short_ema,
            long_ema,
        )
        return self.training_metrics

    def update_market_data(self, row: pd.Series):
        price = float(row.get("close", 0.0))
        if price <= 0:
            return
        self.current_price = price
        self.price_history.append(price)

    def predict(self, steps: int = 1) -> List[float]:
        if not self.is_fitted:
            raise ValueError("Strategy must be fitted before prediction")
        price = self.current_price if self.current_price is not None else 0.0
        return [price for _ in range(steps)]

    def generate_signals(
        self,
        current_price: float,
        predicted_price: float,
        threshold: float = 0.0,
    ) -> str:
        if len(self.price_history) < self.long_window:
            return "HOLD"

        short_ema = self._calculate_short_ema()
        long_ema = self._calculate_long_ema()
        if long_ema == 0:
            return "HOLD"

        diff_pct = (short_ema - long_ema) / long_ema
        if diff_pct > threshold:
            return "BUY"
        if diff_pct < -threshold:
            return "SELL"
        return "HOLD"

    def get_parameters(self) -> Dict[str, Any]:
        params = super().get_parameters()
        params.update({
            "short_window": self.short_window,
            "long_window": self.long_window,
        })
        return params

    def _calculate_short_ema(self) -> float:
        data = list(self.price_history)[-self.long_window:]
        return _ema_value(data, self.short_window)

    def _calculate_long_ema(self) -> float:
        data = list(self.price_history)[-self.long_window:]
        return _ema_value(data, self.long_window)
