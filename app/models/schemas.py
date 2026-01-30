from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class FetchDataRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL')")
    timeframe: str = Field("1d", description="Timeframe: 1m, 5m, 15m, 1h, 1d")
    candles: int = Field(1000, description="Number of candles to fetch", ge=1, le=10000)


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
    strategy: str = Field("ARIMA", description="Strategy name")
    timeframe: str = Field("1d", description="Timeframe")
    candles: int = Field(1000, description="Number of candles to use", ge=1, le=10000)
    initial_capital: float = Field(10000.0, description="Initial capital", gt=0)

    # Moving Average strategy parameters
    threshold: float = Field(0.01, description="Signal threshold (1% default)", ge=0)
    short_window: Optional[int] = Field(
        20,
        description="Short moving average window (Moving Average strategy)",
        gt=0
    )
    long_window: Optional[int] = Field(
        50,
        description="Long moving average window (Moving Average strategy)",
        gt=0
    )

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
    strategy: str = Field(..., description="Strategy name (ARIMA, MOVING_AVERAGE, LSTM)")
    # ARIMA-specific parameters
    arima_order: Optional[List[int]] = Field(None, description="ARIMA order [p, d, q]")
    arima_update_frequency: Optional[int] = Field(50, description="ARIMA update frequency", ge=0, le=500)
    arima_use_confidence_intervals: Optional[bool] = Field(True, description="Use ARIMA confidence intervals")
    arima_confidence_level: Optional[float] = Field(0.95, description="ARIMA confidence level", ge=0.5, le=0.99)
    # Moving Average-specific parameters
    short_window: Optional[int] = Field(None, description="Short MA window", gt=0)
    long_window: Optional[int] = Field(None, description="Long MA window", gt=0)
    # LSTM-specific parameters
    lstm_sequence_length: Optional[int] = Field(None, description="LSTM sequence length", gt=0, le=200)
    lstm_units: Optional[List[int]] = Field(None, description="LSTM units per layer")
    lstm_dropout_rate: Optional[float] = Field(None, description="LSTM dropout rate", ge=0.0, le=0.5)
    lstm_epochs: Optional[int] = Field(None, description="LSTM epochs", gt=0, le=200)
    lstm_batch_size: Optional[int] = Field(None, description="LSTM batch size", gt=0, le=256)
    lstm_learning_rate: Optional[float] = Field(None, description="LSTM learning rate", gt=0.0, le=0.1)
    lstm_validation_split: Optional[float] = Field(None, description="LSTM validation split", ge=0.0, le=0.5)


class CompareAlgoRequest(BaseModel):
    algorithm1: AlgorithmConfig = Field(..., description="First algorithm configuration")
    algorithm2: AlgorithmConfig = Field(..., description="Second algorithm configuration")
    tickers: List[str] = Field(..., description="List of ticker symbols to test", min_length=1)
    timeframes: List[str] = Field(..., description="List of timeframes to test", min_length=1)
    candles: List[int] = Field(..., description="List of candle counts to test", min_length=1)
    # Common backtest parameters
    initial_capital: float = Field(10000.0, description="Initial capital", gt=0)
    threshold: float = Field(0.01, description="Signal threshold", ge=0)
    use_options: Optional[bool] = Field(False, description="Use options trading")
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
