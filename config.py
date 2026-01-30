"""Configuration settings for Kairos backend."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    DATABASE_URL: str = "postgresql://postgres:su@localhost:5432/kairos"
    
    # API
    API_TITLE: str = "Kairos Trading Backtest API"
    API_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    
    # Data Settings
    DEFAULT_CANDLES: int = 1000
    DEFAULT_TIMEFRAME: str = "1d"
    
    
    # Backtesting
    DEFAULT_INITIAL_CAPITAL: float = 10000.0
    TRANSACTION_COST: float = 0.001  # 0.1% per transaction
    
    # Caching
    CACHE_ENABLED: bool = True
    # Max candles to keep per (stock, timeframe); older rows are deleted when count exceeds this
    CACHE_MAX_CANDLES: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

