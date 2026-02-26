"""Pytest configuration and fixtures."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.models.database import Base, Stock, PriceData
from app.database import get_db
from app.services.data_fetcher import DataFetcher
from app.services.data_processor import DataProcessor
from app.strategies.moving_average import MovingAverageStrategy


@pytest.fixture(scope="function")
def db_session():
    """Create a test database session."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    # Generate random walk price data
    prices = 100 + np.cumsum(np.random.randn(100) * 2)
    
    data = {
        'timestamp': dates,
        'open': prices + np.random.randn(100) * 0.5,
        'high': prices + np.abs(np.random.randn(100) * 1),
        'low': prices - np.abs(np.random.randn(100) * 1),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 100)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_stock(db_session):
    """Create a sample stock in the database."""
    stock = Stock(ticker="TEST", name="Test Stock")
    db_session.add(stock)
    db_session.commit()
    db_session.refresh(stock)
    return stock


@pytest.fixture
def data_fetcher(db_session):
    """Create a DataFetcher instance."""
    return DataFetcher(db_session)


@pytest.fixture
def data_processor():
    """Create a DataProcessor instance."""
    return DataProcessor()


@pytest.fixture
def moving_average_strategy():
    """Create a MovingAverageStrategy instance."""
    return MovingAverageStrategy(short_window=20, long_window=50)

