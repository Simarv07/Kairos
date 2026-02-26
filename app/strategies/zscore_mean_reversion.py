import logging
from collections import deque
from typing import Dict, Any, List, Optional

import pandas as pd

from app.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


# Z-Score Mean Reversion: BUY when price is far below mean (negative z-score), SELL when far above
def _zscore(price: float, prices: List[float]) -> float:
    if not prices or len(prices) < 2:
        return 0.0
    mean = sum(prices) / len(prices)
    variance = sum((x - mean) ** 2 for x in prices) / len(prices)
    std = variance ** 0.5
    if std == 0:
        return 0.0
    return (price - mean) / std


class ZScoreMeanReversionStrategy(BaseStrategy):
    def __init__(
        self,
        lookback: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.0,
        **kwargs
    ):
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer")
        if entry_z <= 0:
            raise ValueError("entry_z must be positive (use abs z for entries)")

        super().__init__(
            lookback=lookback,
            entry_z=entry_z,
            exit_z=exit_z,
            **kwargs
        )
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.price_history: deque = deque(maxlen=lookback + 50)
        self.current_price: Optional[float] = None
        self.training_metrics: Dict[str, Any] = {}
        self.is_fitted = False

    def fit(self, train_data: pd.DataFrame) -> Dict[str, Any]:
        if train_data.empty or "close" not in train_data.columns:
            raise ValueError("Training data must include 'close' prices")

        closes = train_data["close"].dropna().tolist()
        if len(closes) < self.lookback:
            raise ValueError(
                f"Insufficient data for Z-Score (need at least {self.lookback} closes)"
            )

        for price in closes[-self.lookback - 50 :]:
            self.price_history.append(price)
        self.current_price = self.price_history[-1]

        z = _zscore(self.current_price, list(self.price_history)[-self.lookback:])
        self.training_metrics = {
            "lookback": self.lookback,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
            "initial_zscore": z,
        }
        self.is_fitted = True
        logger.info("ZScoreMeanReversionStrategy initialized with z=%.4f", z)
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
        if len(self.price_history) < self.lookback:
            return "HOLD"

        window = list(self.price_history)[-self.lookback:]
        z = _zscore(current_price, window)
        # Mean reversion: BUY when price is low (z < -entry_z), SELL when high (z > entry_z)
        # If threshold is passed, use it as extra buffer (e.g. entry_z + threshold)
        entry = self.entry_z + (threshold if threshold else 0)
        if z < -entry:
            return "BUY"
        if z > entry:
            return "SELL"
        return "HOLD"

    def get_parameters(self) -> Dict[str, Any]:
        params = super().get_parameters()
        params.update({
            "lookback": self.lookback,
            "entry_z": self.entry_z,
            "exit_z": self.exit_z,
        })
        return params
