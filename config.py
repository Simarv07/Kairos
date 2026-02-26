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
    # Max candles per (stock, timeframe): API and DB are capped at this; DB is rebalanced (oldest removed) when new data is stored
    CACHE_MAX_CANDLES: int = 1000

    # WhatsApp (Twilio)
    # Note: Twilio WhatsApp requires the `whatsapp:` prefix, e.g. "whatsapp:+15551234567"
    TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_WHATSAPP_FROM: str = os.getenv("TWILIO_WHATSAPP_FROM")
    WHATSAPP_DEFAULT_TO: str = os.getenv("WHATSAPP_DEFAULT_TO")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

