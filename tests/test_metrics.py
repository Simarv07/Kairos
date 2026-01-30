"""Tests for metrics calculation."""
import pytest
from app.services.metrics import calculate_prediction_metrics, calculate_backtest_metrics


def test_calculate_prediction_metrics():
    """Test prediction metrics calculation."""
    actual = [100, 101, 102, 103, 104]
    predicted = [100.5, 101.2, 101.8, 103.1, 104.2]
    
    metrics = calculate_prediction_metrics(actual, predicted)
    
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'mape' in metrics
    assert metrics['mae'] >= 0
    assert metrics['rmse'] >= 0


def test_calculate_backtest_metrics():
    """Test backtest metrics calculation."""
    trades = [
        {'profit_loss': 100, 'entry_price': 100, 'exit_price': 110},
        {'profit_loss': -50, 'entry_price': 100, 'exit_price': 95},
        {'profit_loss': 200, 'entry_price': 100, 'exit_price': 120},
    ]
    
    portfolio_values = [10000, 10100, 10050, 10250]
    
    metrics = calculate_backtest_metrics(
        trades,
        initial_capital=10000.0,
        final_value=10250.0,
        portfolio_values=portfolio_values
    )
    
    assert 'total_return' in metrics
    assert 'win_rate' in metrics
    assert 'num_trades' in metrics
    assert metrics['num_trades'] == 3
    assert metrics['win_rate'] > 66 and metrics['win_rate'] < 67  # 2 out of 3 trades are winners

