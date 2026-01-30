import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
from scipy.stats import norm

logger = logging.getLogger(__name__)


# Calculates historical volatility from price series, returns annualized volatility
def calculate_historical_volatility(prices: np.ndarray, window: int = 30) -> float:
    if len(prices) < 2:
        return 0.20  # Default 20% volatility
    
    # Calculate returns
    returns = np.diff(np.log(prices))
    
    # Use rolling window if we have enough data
    if len(returns) >= window:
        returns = returns[-window:]
    
    # Calculate standard deviation of returns
    std_dev = np.std(returns)
    
    # Annualize (assuming daily data, 252 trading days)
    annualized_vol = std_dev * np.sqrt(252)
    
    # Cap volatility at reasonable bounds
    annualized_vol = max(0.05, min(2.0, annualized_vol))
    
    return float(annualized_vol)


def black_scholes_call(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    # Calculates Black-Scholes call option price
    if T <= 0:
        return max(0, S - K)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return max(0.0, call_price)


def black_scholes_put(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float
) -> float:
    # Calculates Black-Scholes put option price using put-call parity
    if T <= 0:
        return max(0, K - S)
    
    # Use put-call parity
    call_price = black_scholes_call(S, K, T, r, sigma)
    put_price = call_price - S + K * np.exp(-r * T)
    
    return max(0.0, put_price)


def calculate_option_price(
    option_type: str,
    stock_price: float,
    strike_price: float,
    time_to_expiration: float,
    risk_free_rate: float = 0.05,
    volatility: Optional[float] = None,
    historical_prices: Optional[np.ndarray] = None
) -> float:
    # Calculates option price using Black-Scholes model with historical volatility
    # Calculate volatility if not provided
    if volatility is None:
        if historical_prices is not None and len(historical_prices) > 1:
            volatility = calculate_historical_volatility(historical_prices)
        else:
            volatility = 0.20  # Default 20% volatility
    
    option_type = option_type.upper()
    
    if option_type == 'CALL':
        return black_scholes_call(
            stock_price, strike_price, time_to_expiration,
            risk_free_rate, volatility
        )
    elif option_type == 'PUT':
        return black_scholes_put(
            stock_price, strike_price, time_to_expiration,
            risk_free_rate, volatility
        )
    else:
        raise ValueError(f"Invalid option type: {option_type}. Must be 'CALL' or 'PUT'")


# Calculates intrinsic value of an option (CALL or PUT)
def calculate_intrinsic_value(option_type: str, stock_price: float, strike_price: float) -> float:
    option_type = option_type.upper()
    
    if option_type == 'CALL':
        return max(0.0, stock_price - strike_price)
    elif option_type == 'PUT':
        return max(0.0, strike_price - stock_price)
    else:
        raise ValueError(f"Invalid option type: {option_type}")


def select_strike_price(
    stock_price: float,
    strike_selection: str = 'ATM',
    moneyness: float = 0.0
) -> float:
    # Selects strike price based on strategy (ATM, ITM, or OTM)
    strike_selection = strike_selection.upper()
    
    if strike_selection == 'ATM':
        # Round to nearest $5 for cleaner strikes
        return round(stock_price / 5) * 5
    elif strike_selection == 'ITM':
        # In-the-money: lower strike for calls, higher for puts
        # For calls: strike = price * (1 - moneyness)
        return round(stock_price * (1 - moneyness) / 5) * 5
    elif strike_selection == 'OTM':
        # Out-of-the-money: higher strike for calls, lower for puts
        # For calls: strike = price * (1 + moneyness)
        return round(stock_price * (1 + moneyness) / 5) * 5
    else:
        # Default to ATM
        return round(stock_price / 5) * 5

