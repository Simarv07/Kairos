"""Tests for backtester."""
import pytest
import pandas as pd
import numpy as np
from app.services.backtester import Backtester
from app.strategies.moving_average import MovingAverageStrategy
from app.strategies.ema import EMAStrategy
from app.strategies.macd import MACDStrategy
from app.strategies.rsi import RSIStrategy
from app.strategies.zscore_mean_reversion import ZScoreMeanReversionStrategy


def test_backtester_init():
    """Test backtester initialization."""
    backtester = Backtester(initial_capital=10000.0)
    assert backtester.initial_capital == 10000.0
    assert backtester.cash == 10000.0
    assert backtester.position == 0


def test_backtester_run(moving_average_strategy, sample_price_data):
    """Test backtest execution."""
    # Split data
    train_df = sample_price_data.iloc[:80]
    test_df = sample_price_data.iloc[80:]
    
    # Fit strategy
    moving_average_strategy.fit(train_df)
    
    # Run backtest
    backtester = Backtester(initial_capital=10000.0)
    results = backtester.run(moving_average_strategy, test_df, threshold=0.01)
    
    assert 'metrics' in results
    assert 'total_return' in results['metrics']
    assert 'num_trades' in results['metrics']
    assert 'portfolio_values' in results


def test_calculate_portfolio_value():
    """Test portfolio value calculation."""
    backtester = Backtester(initial_capital=10000.0)
    backtester.cash = 5000.0
    backtester.position = 50
    backtester.entry_price = 100.0
    
    value = backtester._calculate_portfolio_value(110.0)
    assert value == 5000.0 + (50 * 110.0)


def test_execute_trade():
    """Test trade execution."""
    backtester = Backtester(initial_capital=10000.0)
    
    # Execute BUY
    backtester._execute_trade('BUY', 100.0, '2023-01-01')
    assert backtester.position > 0
    assert backtester.cash < backtester.initial_capital
    assert len(backtester.trades) == 0 # Check if the trade is not executed
    
    # Execute SELL
    initial_position = backtester.position
    backtester._execute_trade('SELL', 110.0, '2023-01-02')
    assert backtester.position == 0
    assert len(backtester.trades) == 1  # Check if the trade is executed


def test_ema_strategy_backtest(sample_price_data):
    """Test EMA strategy can be fitted and backtested."""
    strategy = EMAStrategy(short_window=12, long_window=26)
    train_df = sample_price_data.iloc[:80]
    test_df = sample_price_data.iloc[80:]
    strategy.fit(train_df)
    backtester = Backtester(initial_capital=10000.0)
    results = backtester.run(strategy, test_df, threshold=0.01)
    assert 'metrics' in results
    assert 'total_return' in results['metrics']
    assert 'portfolio_values' in results


def test_macd_strategy_backtest(sample_price_data):
    """Test MACD strategy can be fitted and backtested."""
    strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
    train_df = sample_price_data.iloc[:80]
    test_df = sample_price_data.iloc[80:]
    strategy.fit(train_df)
    backtester = Backtester(initial_capital=10000.0)
    results = backtester.run(strategy, test_df, threshold=0.0)
    assert 'metrics' in results
    assert 'total_return' in results['metrics']
    assert 'portfolio_values' in results


def test_rsi_strategy_backtest(sample_price_data):
    """Test RSI strategy can be fitted and backtested."""
    strategy = RSIStrategy(period=14, oversold=30.0, overbought=70.0)
    train_df = sample_price_data.iloc[:80]
    test_df = sample_price_data.iloc[80:]
    strategy.fit(train_df)
    backtester = Backtester(initial_capital=10000.0)
    results = backtester.run(strategy, test_df)
    assert 'metrics' in results
    assert 'total_return' in results['metrics']
    assert 'portfolio_values' in results


def test_zscore_mean_reversion_backtest(sample_price_data):
    """Test Z-Score Mean Reversion strategy can be fitted and backtested."""
    strategy = ZScoreMeanReversionStrategy(lookback=20, entry_z=2.0)
    train_df = sample_price_data.iloc[:80]
    test_df = sample_price_data.iloc[80:]
    strategy.fit(train_df)
    backtester = Backtester(initial_capital=10000.0)
    results = backtester.run(strategy, test_df)
    assert 'metrics' in results
    assert 'total_return' in results['metrics']
    assert 'portfolio_values' in results

