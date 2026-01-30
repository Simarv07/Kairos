from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


# Stock ticker information
class Stock(Base):
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String(10), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    price_data = relationship("PriceData", back_populates="stock", cascade="all, delete-orphan")
    backtests = relationship("Backtest", back_populates="stock", cascade="all, delete-orphan")


# Historical price data (OHLCV)
class PriceData(Base):
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)  # 1m, 5m, 15m, 1h, 1d
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False, index=True)
    volume = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="price_data")
    
    # Unique constraint on stock_id, timestamp, timeframe
    __table_args__ = (
        {"sqlite_autoincrement": True},
    )


# Trading strategy definitions
class Strategy(Base):
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    parameters = Column(JSON, nullable=True)  # Store strategy-specific parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    backtests = relationship("Backtest", back_populates="strategy")


# Backtest execution results
class Backtest(Base):
    __tablename__ = "backtests"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    strategy_id = Column(Integer, ForeignKey("strategies.id"), nullable=False)
    
    # Parameters
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Results
    total_return = Column(Float, nullable=True)
    win_rate = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    num_trades = Column(Integer, nullable=True)
    avg_gain_per_trade = Column(Float, nullable=True)
    sharpe_ratio = Column(Float, nullable=True)
    
    # Metrics (stored as JSON for flexibility)
    metrics = Column(JSON, nullable=True)  # MAE, RMSE, MAPE, etc.
    model_parameters = Column(JSON, nullable=True)  # ARIMA order, etc.
    
    # Execution info
    status = Column(String(20), default="pending")  # pending, running, completed, failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    stock = relationship("Stock", back_populates="backtests")
    strategy = relationship("Strategy", back_populates="backtests")

