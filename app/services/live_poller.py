"""
Live price polling:
- Fetch data at an interval based on timeframe
- Keep a rolling window of candles (max 1000, aligned with cache)
- Evaluate the strategy on each new candle
- Send WhatsApp notifications on BUY/SELL signals (equity or options-style)
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from app.database import get_db_context
from app.services.data_fetcher import DataFetcher
from app.services.data_processor import DataProcessor
from app.strategies.base import BaseStrategy
from app.strategies.ema import EMAStrategy
from app.strategies.macd import MACDStrategy
from app.strategies.moving_average import MovingAverageStrategy
from app.strategies.rsi import RSIStrategy
from app.strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy
from app.services.options_pricing import select_strike_price

logger = logging.getLogger(__name__)


# Poll interval in seconds per timeframe
POLL_INTERVAL_SECONDS: Dict[str, int] = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "1d": 24 * 60 * 60,
}

# Align with cache max candles
ROLLING_CANDLES = 1000


def _create_strategy(
    strategy_name: str,
    *,
    short_window: Optional[int] = None,
    long_window: Optional[int] = None,
    fast_period: Optional[int] = None,
    slow_period: Optional[int] = None,
    signal_period: Optional[int] = None,
    rsi_period: Optional[int] = None,
    oversold: Optional[float] = None,
    overbought: Optional[float] = None,
    lookback: Optional[int] = None,
    entry_z: Optional[float] = None,
) -> BaseStrategy:
    """Build strategy instance from name and optional parameters."""
    name = (strategy_name or "").upper()

    if name in ("MA", "MOVING_AVERAGE"):
        return MovingAverageStrategy(
            short_window=short_window or 20,
            long_window=long_window or 50,
        )
    if name == "EMA":
        return EMAStrategy(
            short_window=short_window or 12,
            long_window=long_window or 26,
        )
    if name == "MACD":
        return MACDStrategy(
            fast_period=fast_period or 12,
            slow_period=slow_period or 26,
            signal_period=signal_period or 9,
        )
    if name == "RSI":
        return RSIStrategy(
            period=rsi_period or 14,
            oversold=oversold or 30.0,
            overbought=overbought or 70.0,
        )
    if name in ("ZSCORE", "Z_SCORE", "ZSCORE_MEAN_REVERSION"):
        return ZScoreMeanReversionStrategy(
            lookback=lookback or 20,
            entry_z=entry_z or 2.0,
        )

    raise ValueError(f"Unknown strategy: {strategy_name}")


@dataclass
class LivePollConfig:
    """Configuration for a single live poll job."""

    poll_id: str
    ticker: str
    timeframe: str
    strategy_name: str
    whatsapp_to: str
    threshold: float = 0.01

    # Strategy params (optional)
    short_window: Optional[int] = None
    long_window: Optional[int] = None
    fast_period: Optional[int] = None
    slow_period: Optional[int] = None
    signal_period: Optional[int] = None
    rsi_period: Optional[int] = None
    oversold: Optional[float] = None
    overbought: Optional[float] = None
    lookback: Optional[int] = None
    entry_z: Optional[float] = None

    # Options: use_options, option_type (CALL/PUT/BOTH), leap_expiration_days, strike_selection, risk_free_rate
    use_options: bool = False
    option_type: str = "BOTH"
    leap_expiration_days: int = 730
    strike_selection: str = "ATM"
    risk_free_rate: float = 0.05


@dataclass
class LivePollState:
    """Runtime state for a live poll job."""

    last_candle_timestamp: Optional[Any] = None
    last_signal: Optional[str] = None
    last_eval_at: Optional[datetime] = None
    last_error: Optional[str] = None
    candles_count: int = 0


class LivePoller:
    """Manages live price polling jobs: start, stop, status."""

    def __init__(self, send_whatsapp: Callable[[str, str], Any]) -> None:
        """
        send_whatsapp(to: str, body: str) is called when a trade signal should be notified.
        It can be async; we will await it if it's a coroutine.
        """
        self._send_whatsapp = send_whatsapp
        self._jobs: Dict[str, Tuple[LivePollConfig, LivePollState, Optional[asyncio.Task]]] = {}
        self._lock = asyncio.Lock()

    def _poll_interval_seconds(self, timeframe: str) -> int:
        return POLL_INTERVAL_SECONDS.get(timeframe, 60 * 60)

    async def _notify_trade(self, to: str, body: str) -> None:
        fn = self._send_whatsapp(to, body)
        if asyncio.iscoroutine(fn):
            await fn

    async def _run_poll_loop(self, poll_id: str) -> None:
        async with self._lock:
            if poll_id not in self._jobs:
                return
            config, state, _ = self._jobs[poll_id]
            # Keep task reference in _jobs so status() can expose running=True

        interval = self._poll_interval_seconds(config.timeframe)
        processor = DataProcessor()

        while True:
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break

            async with self._lock:
                if poll_id not in self._jobs:
                    break
                config, state, _ = self._jobs[poll_id]

            try:
                # Fetch rolling window from DB/yfinance
                with get_db_context() as db:
                    fetcher = DataFetcher(db)
                    df = fetcher.fetch_data(
                        ticker=config.ticker,
                        timeframe=config.timeframe,
                        candles=ROLLING_CANDLES,
                    )

                if df.empty or len(df) < 50:
                    state.last_error = "Insufficient data"
                    continue

                df = processor.preprocess_data(df)
                state.candles_count = len(df)
                last_row = df.iloc[-1]
                current_ts = last_row.get("timestamp")
                current_price = float(last_row.get("close", 0))

                # Detect new candle
                if state.last_candle_timestamp is not None and current_ts == state.last_candle_timestamp:
                    continue

                state.last_candle_timestamp = current_ts
                state.last_eval_at = datetime.utcnow()
                state.last_error = None

                # Build strategy and fit on rolling window
                strategy = _create_strategy(
                    config.strategy_name,
                    short_window=config.short_window,
                    long_window=config.long_window,
                    fast_period=config.fast_period,
                    slow_period=config.slow_period,
                    signal_period=config.signal_period,
                    rsi_period=config.rsi_period,
                    oversold=config.oversold,
                    overbought=config.overbought,
                    lookback=config.lookback,
                    entry_z=config.entry_z,
                )
                strategy.fit(df)

                # One-step evaluation: update state with last row, then generate signal
                if hasattr(strategy, "update_market_data"):
                    strategy.update_market_data(last_row)
                try:
                    pred = strategy.predict(steps=1)
                    predicted_price = pred[0] if pred else current_price
                except Exception:
                    predicted_price = current_price

                signal = strategy.generate_signals(
                    current_price, predicted_price, config.threshold
                )
                state.last_signal = signal

                if signal in ("BUY", "SELL"):
                    ts_str = str(current_ts) if current_ts else "N/A"

                    # Options mapping: CALL on BUY, PUT on SELL when BOTH; or forced CALL/PUT
                    option_type_config = (config.option_type or "BOTH").upper()
                    if option_type_config not in ("CALL", "PUT", "BOTH"):
                        option_type_config = "BOTH"

                    option_label: Optional[str] = None
                    if config.use_options:
                        if signal == "BUY" and option_type_config in ("CALL", "BOTH"):
                            option_label = "CALL"
                        elif signal == "SELL" and option_type_config in ("PUT", "BOTH"):
                            option_label = "PUT"
                        elif option_type_config == "CALL" and signal == "BUY":
                            option_label = "CALL"
                        elif option_type_config == "PUT" and signal == "SELL":
                            option_label = "PUT"

                    body_lines = [
                        "Kairos Trade Signal",
                        f"Ticker: {config.ticker}",
                        f"Timeframe: {config.timeframe}",
                        f"Signal: {signal}",
                        f"Price: {current_price:.2f}",
                        f"Candle time: {ts_str}",
                        f"Strategy: {config.strategy_name}",
                    ]

                    if option_label:
                        strike = select_strike_price(
                            current_price,
                            config.strike_selection or "ATM",
                        )
                        body_lines.append(f"Option: {option_label}")
                        body_lines.append(f"Strike: ${strike:.2f} ({config.strike_selection or 'ATM'})")
                        body_lines.append(f"Expiration: {config.leap_expiration_days} days (LEAP)")

                    body = "\n".join(body_lines)
                    try:
                        await self._notify_trade(config.whatsapp_to, body)
                    except Exception as e:
                        logger.exception("WhatsApp notification failed: %s", e)
                        state.last_error = str(e)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Live poll iteration failed for %s: %s", poll_id, e)
                async with self._lock:
                    if poll_id in self._jobs:
                        _, state, _ = self._jobs[poll_id]
                        state.last_error = str(e)

        logger.info("Live poll loop exited for %s", poll_id)

    async def start(self, config: LivePollConfig) -> str:
        """Start a new poll job. Returns poll_id."""
        poll_id = config.poll_id or str(uuid.uuid4())
        async with self._lock:
            if poll_id in self._jobs:
                raise ValueError(f"Poll already running: {poll_id}")

            state = LivePollState()
            task = asyncio.create_task(
                self._run_poll_loop(poll_id),
                name=f"live_poll_{config.ticker}_{config.timeframe}",
            )
            self._jobs[poll_id] = (config, state, task)

        logger.info(
            "Started live poll %s for %s %s (strategy=%s, interval=%ss)",
            poll_id,
            config.ticker,
            config.timeframe,
            config.strategy_name,
            self._poll_interval_seconds(config.timeframe),
        )
        return poll_id

    async def stop(self, poll_id: str) -> Optional[LivePollConfig]:
        """Stop a poll job. Returns the poll config if it was running, else None."""
        async with self._lock:
            if poll_id not in self._jobs:
                return None
            config, state, task = self._jobs.pop(poll_id)

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        logger.info("Stopped live poll %s", poll_id)
        return config

    async def status(self) -> list:
        """Return list of active poll jobs with their state."""
        result = []
        async with self._lock:
            for poll_id, (config, state, task) in list(self._jobs.items()):
                running = bool(task is not None and not task.done())
                result.append(
                    {
                        "poll_id": poll_id,
                        "ticker": config.ticker,
                        "timeframe": config.timeframe,
                        "strategy": config.strategy_name,
                        "whatsapp_to": config.whatsapp_to,
                        "last_signal": state.last_signal,
                        "last_eval_at": state.last_eval_at.isoformat()
                        if state.last_eval_at
                        else None,
                        "last_error": state.last_error,
                        "candles_count": state.candles_count,
                        "running": running,
                    }
                )
        return result

    def is_running(self, poll_id: str) -> bool:
        if poll_id not in self._jobs:
            return False
        _, _, task = self._jobs[poll_id]
        return bool(task is not None and not task.done())


def _default_whatsapp_sender():
    """Default sender: no-op unless Twilio is configured; otherwise send via Twilio."""

    def _noop(to: str, body: str) -> None:
        logger.warning("WhatsApp not configured; skipping notification to %s", to)

    async def _send(to: str, body: str) -> None:
        from config import settings
        from app.services.whatsapp import TwilioWhatsAppClient

        if (
            not settings.TWILIO_ACCOUNT_SID
            or not settings.TWILIO_AUTH_TOKEN
            or not settings.TWILIO_WHATSAPP_FROM
        ):
            _noop(to, body)
            return

        client = TwilioWhatsAppClient(
            account_sid=settings.TWILIO_ACCOUNT_SID,
            auth_token=settings.TWILIO_AUTH_TOKEN,
            whatsapp_from=settings.TWILIO_WHATSAPP_FROM,
        )
        await client.send_message(to=to, body=body)

    return _send


# Singleton used by API and shutdown hook
live_poller = LivePoller(send_whatsapp=_default_whatsapp_sender())

