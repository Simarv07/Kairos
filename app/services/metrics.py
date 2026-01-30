import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_prediction_metrics(
    actual: List[float],
    predicted: List[float]
) -> Dict[str, float]:
    # Calculates prediction accuracy metrics (MAE, RMSE, MAPE)
    actual = np.array(actual)
    predicted = np.array(predicted)
    
    # Remove any NaN or inf values
    mask = np.isfinite(actual) & np.isfinite(predicted)
    actual = actual[mask]
    predicted = predicted[mask]
    
    if len(actual) == 0:
        return {
            'mae': 0.0,
            'rmse': 0.0,
            'mape': 0.0
        }
    
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-10))) * 100
    
    return {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape)
    }


def calculate_backtest_metrics(
    trades: List[Dict[str, Any]],
    initial_capital: float,
    final_value: float,
    portfolio_values: List[float]
) -> Dict[str, Any]:
    # Calculates backtesting performance metrics (return, win rate, drawdown, Sharpe ratio)
    if not trades:
        return {
            'total_return': 0.0,
            'win_rate': 0.0,
            'max_drawdown': 0.0,
            'num_trades': 0,
            'avg_gain_per_trade': 0.0,
            'sharpe_ratio': 0.0
        }
    
    # Total return
    total_return = ((final_value - initial_capital) / initial_capital) * 100
    
    # Win rate
    winning_trades = [t for t in trades if t.get('profit_loss', 0) > 0]
    win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0.0
    
    # Average gain per trade
    profits = [t.get('profit_loss', 0) for t in trades]
    avg_gain_per_trade = np.mean(profits) if profits else 0.0
    
    # Maximum drawdown
    portfolio_array = np.array(portfolio_values)
    if len(portfolio_array) > 0:
        running_max = np.maximum.accumulate(portfolio_array)
        drawdown = (portfolio_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdown)) * 100 if len(drawdown) > 0 else 0.0
    else:
        max_drawdown = 0.0
    
    # Sharpe ratio (simplified - using returns)
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)  # Annualized
        else:
            sharpe_ratio = 0.0
    else:
        sharpe_ratio = 0.0
    
    return {
        'total_return': float(total_return),
        'win_rate': float(win_rate),
        'max_drawdown': float(max_drawdown),
        'num_trades': len(trades),
        'avg_gain_per_trade': float(avg_gain_per_trade),
        'sharpe_ratio': float(sharpe_ratio)
    }

