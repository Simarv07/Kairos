from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator


class FetchDataRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    timeframe: str = Field("1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d")
    candles: int = Field(1000, description="Number of candles to fetch (maximum 1000)", ge=1, le=1000)


class FetchDataResponse(BaseModel):
    success: bool
    message: str
    ticker: str
    timeframe: str
    candles_fetched: int
    cached: bool = False


class TradeDetail(BaseModel):
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    capital_invested: float
    profit_loss: float
    profit_loss_pct: float
    option_type: Optional[str] = None  # 'CALL' or 'PUT' for options
    strike_price: Optional[float] = None
    expiration_date: Optional[str] = None
    contracts: Optional[int] = None
    option_entry_price: Optional[float] = None
    option_exit_price: Optional[float] = None
    expired: Optional[bool] = None


class BacktestRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    strategy: str = Field("MOVING_AVERAGE", description="Strategy name")
    timeframe: str = Field("1d", description="Timeframe")
    candles: int = Field(1000, description="Number of candles to use (maximum 1000)", ge=1, le=1000)
    initial_capital: float = Field(10000.0, description="Initial capital", gt=0)

    # Moving Average / EMA strategy parameters
    threshold: float = Field(0.01, description="Signal threshold (1% default)", ge=0)
    short_window: Optional[int] = Field(
        20,
        description="Short MA/EMA window (Moving Average, EMA strategies)",
        gt=0
    )
    long_window: Optional[int] = Field(
        50,
        description="Long MA/EMA window (Moving Average, EMA strategies)",
        gt=0
    )
    # MACD strategy parameters
    fast_period: Optional[int] = Field(12, description="MACD fast EMA period", gt=0)
    slow_period: Optional[int] = Field(26, description="MACD slow EMA period", gt=0)
    signal_period: Optional[int] = Field(9, description="MACD signal line period", gt=0)
    # RSI strategy parameters
    rsi_period: Optional[int] = Field(14, description="RSI period", gt=0)
    oversold: Optional[float] = Field(30.0, description="RSI oversold level", ge=0, le=100)
    overbought: Optional[float] = Field(70.0, description="RSI overbought level", ge=0, le=100)
    # Z-Score Mean Reversion parameters
    lookback: Optional[int] = Field(20, description="Z-Score lookback window", gt=0)
    entry_z: Optional[float] = Field(2.0, description="Z-Score entry threshold", gt=0)

    # Options trading parameters
    use_options: Optional[bool] = Field(
        False,
        description="Use LEAPs (options) instead of equity trading"
    )
    leap_expiration_days: Optional[int] = Field(
        730,
        description="Days until LEAP expiration (default 730 days = 2 years)",
        gt=30,
        le=1095
    )
    strike_selection: Optional[str] = Field(
        "ATM",
        description="Strike selection: 'ATM' (at-the-money), 'ITM' (in-the-money), 'OTM' (out-of-the-money)"
    )
    risk_free_rate: Optional[float] = Field(
        0.05,
        description="Risk-free interest rate for options pricing (default 5%)",
        ge=0.0,
        le=0.2
    )
    option_type: Optional[str] = Field(
        "CALL",
        description="Options direction: CALL (only calls on BUY), PUT (only puts on SELL), BOTH (calls on BUY, puts on SELL)"
    )


class BacktestResponse(BaseModel):
    backtest_id: int
    status: str
    ticker: str
    strategy: str
    start_date: datetime
    end_date: datetime
    final_value: Optional[float] = None
    profit_loss: Optional[float] = None
    trades: Optional[List[TradeDetail]] = None
    total_return: Optional[float] = None
    win_rate: Optional[float] = None
    max_drawdown: Optional[float] = None
    num_trades: Optional[int] = None
    avg_gain_per_trade: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    model_parameters: Optional[Dict[str, Any]] = None
    message: Optional[str] = None


class BacktestDetailResponse(BacktestResponse):
    initial_capital: float
    timeframe: str
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class AlgorithmConfig(BaseModel):
    strategy: str = Field(..., description="Strategy name: MOVING_AVERAGE, EMA, MACD, RSI, ZSCORE")
    # Moving Average / EMA
    short_window: Optional[int] = Field(None, description="Short MA/EMA window", gt=0)
    long_window: Optional[int] = Field(None, description="Long MA/EMA window", gt=0)
    # MACD
    fast_period: Optional[int] = Field(None, description="MACD fast period", gt=0)
    slow_period: Optional[int] = Field(None, description="MACD slow period", gt=0)
    signal_period: Optional[int] = Field(None, description="MACD signal period", gt=0)
    # RSI
    rsi_period: Optional[int] = Field(None, description="RSI period", gt=0)
    oversold: Optional[float] = Field(None, description="RSI oversold", ge=0, le=100)
    overbought: Optional[float] = Field(None, description="RSI overbought", ge=0, le=100)
    # Z-Score Mean Reversion
    lookback: Optional[int] = Field(None, description="Z-Score lookback", gt=0)
    entry_z: Optional[float] = Field(None, description="Z-Score entry threshold", gt=0)


class CompareAlgoRequest(BaseModel):
    algorithm1: AlgorithmConfig = Field(..., description="First algorithm configuration")
    algorithm2: AlgorithmConfig = Field(..., description="Second algorithm configuration")
    tickers: List[str] = Field(..., description="List of ticker symbols to test", min_length=1)
    timeframes: List[str] = Field(..., description="List of timeframes to test", min_length=1)
    candles: List[int] = Field(..., description="List of candle counts to test (each max 1000)", min_length=1)

    @field_validator("candles")
    @classmethod
    def candles_max_1000(cls, v: List[int]) -> List[int]:
        for c in v:
            if c < 1 or c > 1000:
                raise ValueError("Each candle count must be between 1 and 1000 (maximum 1000 candles)")
        return v
    # Common backtest parameters
    initial_capital: float = Field(10000.0, description="Initial capital", gt=0)
    threshold: float = Field(0.01, description="Signal threshold", ge=0)
    use_options: Optional[bool] = Field(False, description="Use options trading")
    option_type: Optional[str] = Field(
        "CALL",
        description="Options: CALL, PUT, or BOTH (calls on BUY, puts on SELL)"
    )
    leap_expiration_days: Optional[int] = Field(730, description="LEAP expiration days", gt=30, le=1095)
    strike_selection: Optional[str] = Field("ATM", description="Strike selection method")
    risk_free_rate: Optional[float] = Field(0.05, description="Risk-free rate", ge=0.0, le=0.2)


class AlgorithmMetrics(BaseModel):
    total_return: float
    win_rate: float
    max_drawdown: float
    num_trades: int
    avg_gain_per_trade: float
    sharpe_ratio: float
    final_value: float
    profit_loss: float


class ComparisonResult(BaseModel):
    ticker: str
    timeframe: str
    candles: int
    algorithm1_name: str
    algorithm2_name: str
    algorithm1_metrics: AlgorithmMetrics
    algorithm2_metrics: AlgorithmMetrics
    winner: str = Field(..., description="'algorithm1', 'algorithm2', or 'tie'")
    error: Optional[str] = None


class OverallComparison(BaseModel):
    algorithm1_name: str
    algorithm2_name: str
    total_combinations: int
    successful_combinations: int
    algorithm1_wins: int
    algorithm2_wins: int
    ties: int
    algorithm1_avg_metrics: AlgorithmMetrics
    algorithm2_avg_metrics: AlgorithmMetrics
    overall_winner: str = Field(..., description="'algorithm1', 'algorithm2', or 'tie'")


class CompareAlgoResponse(BaseModel):
    results: List[ComparisonResult]
    overall_comparison: OverallComparison
    message: str


# Live polling: start polling for price updates and strategy evaluation; notify via WhatsApp on trade signals
class LivePollRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol")
    timeframe: str = Field("1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d")
    strategy: str = Field("EMA", description="Strategy name: MOVING_AVERAGE, EMA, MACD, RSI, ZSCORE")
    whatsapp_to: str = Field(
        ...,
        description='WhatsApp recipient (e.g. "+15551234567" or "whatsapp:+15551234567") for trade notifications'
    )
    threshold: float = Field(0.01, description="Signal threshold", ge=0)
    short_window: Optional[int] = Field(None, description="Short MA/EMA window", gt=0)
    long_window: Optional[int] = Field(None, description="Long MA/EMA window", gt=0)
    fast_period: Optional[int] = Field(None, description="MACD fast period", gt=0)
    slow_period: Optional[int] = Field(None, description="MACD slow period", gt=0)
    signal_period: Optional[int] = Field(None, description="MACD signal period", gt=0)
    rsi_period: Optional[int] = Field(None, description="RSI period", gt=0)
    oversold: Optional[float] = Field(None, description="RSI oversold", ge=0, le=100)
    overbought: Optional[float] = Field(None, description="RSI overbought", ge=0, le=100)
    lookback: Optional[int] = Field(None, description="Z-Score lookback", gt=0)
    entry_z: Optional[float] = Field(None, description="Z-Score entry threshold", gt=0)
    # Options: use options signals (CALL on BUY, PUT on SELL when option_type is BOTH)
    use_options: bool = Field(
        False,
        description="Include options (CALL/PUT) in trade signals; use option_type to choose CALL, PUT, or BOTH",
    )
    option_type: str = Field(
        "BOTH",
        description="Options: CALL (only on BUY), PUT (only on SELL), BOTH (calls on BUY, puts on SELL)",
    )
    leap_expiration_days: Optional[int] = Field(
        730,
        description="Days until LEAP expiration for options (default 730)",
        gt=30,
        le=1095,
    )
    strike_selection: Optional[str] = Field(
        "ATM",
        description="Strike selection: ATM, ITM, or OTM",
    )
    risk_free_rate: Optional[float] = Field(
        0.05,
        description="Risk-free rate for options (default 5%)",
        ge=0.0,
        le=0.2,
    )


class LivePollResponse(BaseModel):
    success: bool
    poll_id: str
    message: str
    ticker: str
    timeframe: str
    strategy: str
    poll_interval_seconds: int


class LivePollStatusEntry(BaseModel):
    poll_id: str
    ticker: str
    timeframe: str
    strategy: str
    whatsapp_to: str
    last_signal: Optional[str] = None
    last_eval_at: Optional[str] = None
    last_error: Optional[str] = None
    candles_count: int = 0
    running: bool = True


class LivePollStatusResponse(BaseModel):
    polls: List[LivePollStatusEntry]
    message: str


class LivePollStopRequest(BaseModel):
    poll_id: str = Field(..., description="Poll ID returned from POST /live/start")
