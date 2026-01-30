import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from app.strategies.base import BaseStrategy
from app.services.metrics import calculate_backtest_metrics, calculate_prediction_metrics
from app.services.options_pricing import (
    calculate_option_price,
    select_strike_price,
    calculate_intrinsic_value
)
from config import settings

logger = logging.getLogger(__name__)


# Backtesting engine for evaluating trading strategies
class Backtester:
    def __init__(
        self,
        initial_capital: float = None,
        transaction_cost: float = None,
        use_options: bool = False,
        leap_expiration_days: int = 730,
        strike_selection: str = 'ATM',
        risk_free_rate: float = 0.05
    ):
        # Initializes backtester with capital, transaction costs, and options settings
        self.initial_capital = initial_capital or settings.DEFAULT_INITIAL_CAPITAL
        self.transaction_cost = transaction_cost or settings.TRANSACTION_COST
        self.use_options = use_options
        self.leap_expiration_days = leap_expiration_days
        self.strike_selection = strike_selection
        self.risk_free_rate = risk_free_rate
        
        self.cash = self.initial_capital
        self.position = 0  # Number of shares held (equity mode) or contracts (options mode)
        self.entry_price = 0.0
        self.entry_timestamp: Optional[Any] = None
        self.entry_cost: float = 0.0
        
        # Options-specific tracking
        self.option_type: Optional[str] = None  # 'CALL' or 'PUT'
        self.strike_price: float = 0.0
        self.expiration_date: Optional[datetime] = None
        self.option_entry_price: float = 0.0  # Price per share (multiply by 100 for contract)
        
        self.trades: List[Dict[str, Any]] = []
        self.portfolio_values: List[float] = []
        self.signals: List[Dict[str, Any]] = []
        self.price_history: List[float] = []  # For volatility calculation
    
    def run(
        self,
        strategy: BaseStrategy,
        historical_data: pd.DataFrame,
        threshold: float = 0.01
    ) -> Dict[str, Any]:
        # Runs backtest on historical data and returns results with metrics
        if not strategy.is_fitted:
            raise ValueError("Strategy must be fitted before backtesting")
        
        logger.info(f"Starting backtest with initial capital: ${self.initial_capital:,.2f}")
        
        # Reset state
        self.cash = self.initial_capital
        self.position = 0
        self.entry_price = 0.0
        self.entry_timestamp = None
        self.entry_cost = 0.0
        self.option_type = None
        self.strike_price = 0.0
        self.expiration_date = None
        self.option_entry_price = 0.0
        self.trades = []
        self.portfolio_values = []
        self.signals = []
        self.price_history = []
        
        # Process each time step
        for idx in range(len(historical_data)):
            row = historical_data.iloc[idx]
            current_price = row['close']
            timestamp = row.get('timestamp', idx)
            
            # Track price history for volatility calculation
            self.price_history.append(current_price)
            if len(self.price_history) > 252:  # Keep last year
                self.price_history.pop(0)
            
            # Check for option expiration
            if self.use_options and self.position > 0 and self.expiration_date:
                if isinstance(timestamp, datetime) and timestamp >= self.expiration_date:
                    # Option expired - close position at intrinsic value
                    self._handle_option_expiration(current_price, timestamp)
                    continue
            
            # Allow strategies to receive the full row context for stateful calculations
            if hasattr(strategy, "update_market_data"):
                try:
                    strategy.update_market_data(row)
                except Exception as e:
                    logger.debug(f"Strategy data update failed at index {idx}: {str(e)}")
            
            # Generate prediction for next period
            try:
                predictions = strategy.predict(steps=1)
                predicted_price = predictions[0] if predictions else current_price
            except Exception as e:
                logger.warning(f"Prediction failed at index {idx}: {str(e)}")
                predicted_price = current_price
            
            # Generate signal
            signal = strategy.generate_signals(current_price, predicted_price, threshold)
            
            # Execute trade based on signal
            if self.use_options:
                self._execute_option_trade(signal, current_price, timestamp)
            else:
                self._execute_trade(signal, current_price, timestamp)
            
            # Calculate current portfolio value
            portfolio_value = self._calculate_portfolio_value(current_price, timestamp)
            self.portfolio_values.append(portfolio_value)
            
            # Record signal
            self.signals.append({
                'timestamp': timestamp,
                'price': current_price,
                'predicted_price': predicted_price,
                'signal': signal,
                'portfolio_value': portfolio_value
            })
        
        # Close any open position
        if self.position > 0:
            final_price = historical_data.iloc[-1]['close']
            final_timestamp = historical_data.iloc[-1].get('timestamp', len(historical_data) - 1)
            if self.use_options:
                self._close_option_position(final_price, final_timestamp)
            else:
                self._close_position(final_price, final_timestamp)
            final_portfolio_value = self._calculate_portfolio_value(final_price, final_timestamp)
            self.portfolio_values.append(final_portfolio_value)
        else:
            final_portfolio_value = self.portfolio_values[-1] if self.portfolio_values else self.initial_capital
        
        # Calculate metrics
        metrics = calculate_backtest_metrics(
            self.trades,
            self.initial_capital,
            final_portfolio_value,
            self.portfolio_values
        )
        
        # Calculate prediction accuracy if we have enough data
        if len(self.signals) > 1:
            actual_prices = [s['price'] for s in self.signals[1:]]
            predicted_prices = [s['predicted_price'] for s in self.signals[:-1]]
            prediction_metrics = calculate_prediction_metrics(actual_prices, predicted_prices)
        else:
            prediction_metrics = {'mae': 0.0, 'rmse': 0.0, 'mape': 0.0}
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_portfolio_value,
            'trades': self.trades,
            'signals': self.signals,
            'portfolio_values': self.portfolio_values,
            'metrics': {**metrics, **prediction_metrics},
            'start_date': historical_data.iloc[0].get('timestamp', None),
            'end_date': historical_data.iloc[-1].get('timestamp', None)
        }
        
        logger.info(f"Backtest completed. Final value: ${final_portfolio_value:,.2f}, "
                   f"Return: {metrics['total_return']:.2f}%")
        
        return results
    
    # Executes a trade based on signal for equity trading
    def _execute_trade(self, signal: str, price: float, timestamp: Any):
        if signal == 'BUY' and self.position == 0:
            # Buy
            shares = int((self.cash * (1 - self.transaction_cost)) / price)
            if shares > 0:
                cost = shares * price * (1 + self.transaction_cost)
                if cost <= self.cash:
                    self.cash -= cost
                    self.position = shares
                    self.entry_price = price
                    
                    self.entry_timestamp = self._normalize_timestamp(timestamp)
                    self.entry_cost = cost
        
        elif signal == 'SELL' and self.position > 0:
            # Sell
            self._close_position(price, timestamp)
    
    # Closes current equity position and records trade
    def _close_position(self, price: float, timestamp: Any):
        if self.position > 0:
            shares = self.position
            proceeds = shares * price * (1 - self.transaction_cost)
            cost_basis = self.entry_cost or (shares * self.entry_price * (1 + self.transaction_cost))
            profit = proceeds - cost_basis
            capital_invested = cost_basis
            profit_pct = (profit / capital_invested) * 100 if capital_invested else 0.0
            entry_time = self.entry_timestamp or (datetime.utcnow())
            exit_time = self._normalize_timestamp(timestamp)
            
            self.cash += proceeds
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': self.entry_price,
                'exit_price': price,
                'capital_invested': capital_invested,
                'profit_loss': profit,
                'profit_loss_pct': profit_pct
            })
            
            self.position = 0
            self.entry_price = 0.0
            self.entry_timestamp = None
            self.entry_cost = 0.0
    
    # Calculates current portfolio value including options if applicable
    def _calculate_portfolio_value(self, current_price: float, timestamp: Any = None) -> float:
        if self.use_options and self.position > 0:
            # Calculate option value
            if self.expiration_date and timestamp:
                if isinstance(timestamp, datetime):
                    time_to_exp_days = (self.expiration_date - timestamp).days
                else:
                    time_to_exp_days = self.leap_expiration_days
            else:
                time_to_exp_days = self.leap_expiration_days
            
            # Convert days to years for Black-Scholes (which expects years)
            time_to_exp = max(0.0, time_to_exp_days) / 365.0
            
            # Calculate current option price
            historical_prices = np.array(self.price_history) if len(self.price_history) > 1 else None
            option_price = calculate_option_price(
                option_type=self.option_type or 'CALL',
                stock_price=current_price,
                strike_price=self.strike_price,
                time_to_expiration=time_to_exp,
                risk_free_rate=self.risk_free_rate,
                historical_prices=historical_prices
            )
            
            # Contract value = price per share * 100 * number of contracts
            position_value = option_price * 100 * self.position
        else:
            # Equity position
            position_value = self.position * current_price if self.position > 0 else 0
        
        return self.cash + position_value
    
    # Returns trade log as DataFrame
    def get_trade_log(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    # Returns signals log as DataFrame
    def get_signals_log(self) -> pd.DataFrame:
        if not self.signals:
            return pd.DataFrame()
        
        return pd.DataFrame(self.signals)

    # Executes an options trade based on signal
    def _execute_option_trade(self, signal: str, price: float, timestamp: Any):
        if signal == 'BUY' and self.position == 0:
            # Determine option type based on signal direction
            # BUY signal = bullish, so buy CALL
            # SELL signal = bearish, so buy PUT
            option_type = 'CALL'
            
            # Select strike price
            strike = select_strike_price(price, self.strike_selection)
            
            # Calculate expiration date
            if isinstance(timestamp, datetime):
                expiration = timestamp + timedelta(days=self.leap_expiration_days)
            else:
                expiration = datetime.now() + timedelta(days=self.leap_expiration_days)
            
            # Calculate option price (convert days to years for Black-Scholes)
            time_to_exp_years = self.leap_expiration_days / 365.0
            historical_prices = np.array(self.price_history) if len(self.price_history) > 1 else None
            option_price_per_share = calculate_option_price(
                option_type=option_type,
                stock_price=price,
                strike_price=strike,
                time_to_expiration=time_to_exp_years,
                risk_free_rate=self.risk_free_rate,
                historical_prices=historical_prices
            )
            
            # Calculate how many contracts we can afford
            # Contract cost = option_price_per_share * 100 * (1 + transaction_cost)
            contract_cost = option_price_per_share * 100 * (1 + self.transaction_cost)
            
            if contract_cost > 0:
                max_contracts = int(self.cash / contract_cost)
                if max_contracts > 0:
                    total_cost = contract_cost * max_contracts
                    if total_cost <= self.cash:
                        self.cash -= total_cost
                        self.position = max_contracts
                        self.option_type = option_type
                        self.strike_price = strike
                        self.expiration_date = expiration
                        self.option_entry_price = option_price_per_share
                        self.entry_price = price
                        self.entry_timestamp = self._normalize_timestamp(timestamp)
                        self.entry_cost = total_cost
                        
                        logger.info(f"Bought {max_contracts} {option_type} contracts at strike ${strike:.2f}, "
                                  f"expiration {expiration.date()}, cost ${total_cost:.2f}")
        
        elif signal == 'SELL' and self.position > 0:
            # Close option position
            self._close_option_position(price, timestamp)
    
    # Closes current option position and records trade
    def _close_option_position(self, price: float, timestamp: Any):
        if self.position > 0 and self.option_type:
            # Calculate time to expiration
            if self.expiration_date and isinstance(timestamp, datetime):
                time_to_exp_days = (self.expiration_date - timestamp).days
            else:
                time_to_exp_days = self.leap_expiration_days
            
            # Convert days to years for Black-Scholes
            time_to_exp = max(0.0, time_to_exp_days) / 365.0
            
            # Calculate current option price
            historical_prices = np.array(self.price_history) if len(self.price_history) > 1 else None
            current_option_price = calculate_option_price(
                option_type=self.option_type,
                stock_price=price,
                strike_price=self.strike_price,
                time_to_expiration=time_to_exp,
                risk_free_rate=self.risk_free_rate,
                historical_prices=historical_prices
            )
            
            # Calculate proceeds (per contract: price * 100, minus transaction cost)
            proceeds_per_contract = current_option_price * 100 * (1 - self.transaction_cost)
            total_proceeds = proceeds_per_contract * self.position
            
            # Calculate profit/loss
            cost_basis = self.entry_cost
            profit = total_proceeds - cost_basis
            profit_pct = (profit / cost_basis) * 100 if cost_basis > 0 else 0.0
            
            entry_time = self.entry_timestamp or datetime.utcnow()
            exit_time = self._normalize_timestamp(timestamp)
            
            self.cash += total_proceeds
            
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': self.entry_price,
                'exit_price': price,
                'capital_invested': cost_basis,
                'profit_loss': profit,
                'profit_loss_pct': profit_pct,
                'option_type': self.option_type,
                'strike_price': self.strike_price,
                'expiration_date': self.expiration_date.isoformat() if self.expiration_date else None,
                'contracts': self.position,
                'option_entry_price': self.option_entry_price,
                'option_exit_price': current_option_price
            })
            
            logger.info(f"Closed {self.position} {self.option_type} contracts: "
                       f"Profit ${profit:.2f} ({profit_pct:.2f}%)")
            
            # Reset position
            self.position = 0
            self.option_type = None
            self.strike_price = 0.0
            self.expiration_date = None
            self.option_entry_price = 0.0
            self.entry_price = 0.0
            self.entry_timestamp = None
            self.entry_cost = 0.0
    
    # Handles option expiration by closing position at intrinsic value
    def _handle_option_expiration(self, current_price: float, timestamp: Any):
        if self.position > 0 and self.option_type:
            intrinsic_value = calculate_intrinsic_value(self.option_type, current_price, self.strike_price)
            
            # Calculate proceeds at expiration (intrinsic value only, no time value)
            proceeds_per_contract = intrinsic_value * 100 * (1 - self.transaction_cost)
            total_proceeds = proceeds_per_contract * self.position
            
            cost_basis = self.entry_cost
            profit = total_proceeds - cost_basis
            profit_pct = (profit / cost_basis) * 100 if cost_basis > 0 else 0.0
            
            entry_time = self.entry_timestamp or datetime.utcnow()
            exit_time = self._normalize_timestamp(timestamp)
            
            self.cash += total_proceeds
            
            self.trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'entry_price': self.entry_price,
                'exit_price': current_price,
                'capital_invested': cost_basis,
                'profit_loss': profit,
                'profit_loss_pct': profit_pct,
                'option_type': self.option_type,
                'strike_price': self.strike_price,
                'expiration_date': self.expiration_date.isoformat() if self.expiration_date else None,
                'contracts': self.position,
                'option_entry_price': self.option_entry_price,
                'option_exit_price': intrinsic_value,
                'expired': True
            })
            
            logger.info(f"Option expired: {self.position} {self.option_type} contracts, "
                      f"intrinsic value ${intrinsic_value:.2f}, Profit ${profit:.2f} ({profit_pct:.2f}%)")
            
            # Reset position
            self.position = 0
            self.option_type = None
            self.strike_price = 0.0
            self.expiration_date = None
            self.option_entry_price = 0.0
            self.entry_price = 0.0
            self.entry_timestamp = None
            self.entry_cost = 0.0

    # Ensures timestamps are standard datetime objects
    def _normalize_timestamp(self, value: Any) -> Any:
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if isinstance(value, datetime):
            return value
        return value

