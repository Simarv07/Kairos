import logging
from collections import deque
from typing import Dict, Any, List, Optional

import pandas as pd

from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def _ema_from_series(values: List[float], period: int) -> float:
    if not values or len(values) < period:
        return 0.0
    alpha = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = alpha * v + (1 - alpha) * ema
    return float(ema)


# MACD strategy: BUY when MACD line crosses above signal line, SELL when below.
# MACD = EMA_fast(close) - EMA_slow(close), Signal = EMA_signal(MACD)
class MACDStrategy(BaseStrategy):
    def __init__(
        self,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        **kwargs
    ):
        if fast_period <= 0 or slow_period <= 0 or signal_period <= 0:
            raise ValueError("MACD periods must be positive integers")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be smaller than slow_period")

        super().__init__(
            fast_period=fast_period,
            slow_period=slow_period,
            signal_period=signal_period,
            **kwargs
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.price_history: deque = deque(maxlen=slow_period + signal_period + 50)
        self.current_price: Optional[float] = None
        self.training_metrics: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        if train_data.empty or "close" not in train_data.columns:
            raise ValueError("Training data must include 'close' prices")

        closes = train_data["close"].dropna().tolist()
        min_len = self.slow_period + self.signal_period
        if len(closes) < min_len:
            raise ValueError(
                f"Insufficient data for MACD (need at least {min_len} closes)"
            )

        for price in closes[-(min_len + 50) :]:
            self.price_history.append(price)
        self.current_price = self.price_history[-1]

        macd, signal, _ = self._macd_signal_histogram()
        self.training_metrics = {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
            "initial_macd": macd,
            "initial_signal": signal,
        }
        self.is_fitted = True
        logger.info(
            "MACDStrategy initialized with macd=%.4f, signal=%.4f",
            macd,
            signal,
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
        if len(self.price_history) < self.slow_period + self.signal_period:
            return "HOLD"

        macd, signal_line, histogram = self._macd_signal_histogram()
        # Use histogram for crossover: positive = MACD above signal (bullish), negative = bearish
        # threshold as minimum histogram magnitude to avoid noise
        if threshold > 0:
            if histogram > threshold:
                return "BUY"
            if histogram < -threshold:
                return "SELL"
        else:
            if histogram > 0:
                return "BUY"
            if histogram < 0:
                return "SELL"
        return "HOLD"

    def get_parameters(self) -> Dict[str, Any]:
        params = super().get_parameters()
        params.update({
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "signal_period": self.signal_period,
        })
        return params

    def _macd_signal_histogram(self) -> tuple:
        prices = list(self.price_history)
        n = len(prices)
        if n < self.slow_period:
            return 0.0, 0.0, 0.0

        # Build EMAs for each index from slow_period onward
        macd_values = []
        for i in range(self.slow_period, n + 1):
            window = prices[i - self.slow_period : i]
            ema_fast = _ema_from_series(window, self.fast_period)
            ema_slow = _ema_from_series(window, self.slow_period)
            macd_values.append(ema_fast - ema_slow)

        if len(macd_values) < self.signal_period:
            return 0.0, 0.0, 0.0

        signal_line = _ema_from_series(macd_values, self.signal_period)
        macd_line = macd_values[-1]
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
