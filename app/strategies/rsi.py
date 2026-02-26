import logging
from collections import deque
from typing import Dict, Any, List, Optional

import pandas as pd

from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


def _rsi_from_returns(returns: List[float], period: int) -> float:
    """RSI = 100 - 100/(1 + RS), RS = avg_gain / avg_loss over last `period` changes."""
    if len(returns) < period:
        return 50.0  # neutral
    window = returns[-period:]
    gains = [r if r > 0 else 0.0 for r in window]
    losses = [abs(r) if r < 0 else 0.0 for r in window]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


# RSI strategy: BUY when RSI < oversold, SELL when RSI > overbought
class RSIStrategy(BaseStrategy):
    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        **kwargs
    ):
        if period <= 0:
            raise ValueError("RSI period must be a positive integer")
        if not (0 <= oversold < overbought <= 100):
            raise ValueError("oversold must be < overbought and both in [0, 100]")

        super().__init__(
            period=period,
            oversold=oversold,
            overbought=overbought,
            **kwargs
        )
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self.price_history: deque = deque(maxlen=period + 50)
        self.current_price: Optional[float] = None
        self.training_metrics: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        if train_data.empty or "close" not in train_data.columns:
            raise ValueError("Training data must include 'close' prices")

        closes = train_data["close"].dropna().tolist()
        min_len = self.period + 2
        if len(closes) < min_len:
            raise ValueError(
                f"Insufficient data for RSI (need at least {min_len} closes)"
            )

        for price in closes[-(self.period + 50) :]:
            self.price_history.append(price)
        self.current_price = self.price_history[-1]

        rsi_val = self._calculate_rsi()
        self.training_metrics = {
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
            "initial_rsi": rsi_val,
        }
        self.is_fitted = True
        logger.info("RSIStrategy initialized with rsi=%.2f", rsi_val)
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
        if len(self.price_history) < self.period + 2:
            return "HOLD"

        rsi_val = self._calculate_rsi()
        if rsi_val < self.oversold:
            return "BUY"
        if rsi_val > self.overbought:
            return "SELL"
        return "HOLD"

    def get_parameters(self) -> Dict[str, Any]:
        params = super().get_parameters()
        params.update({
            "period": self.period,
            "oversold": self.oversold,
            "overbought": self.overbought,
        })
        return params

    def _calculate_rsi(self) -> float:
        prices = list(self.price_history)
        if len(prices) < 2:
            return 50.0
        returns = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        return _rsi_from_returns(returns, self.period)
