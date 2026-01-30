"""Tests for data fetcher service."""
import pytest
import pandas as pd
from app.services.data_fetcher import DataFetcher
from app.models.database import Stock, PriceData


def test_get_or_create_stock(data_fetcher, db_session):
    """Test stock creation and retrieval."""
    stock = data_fetcher._get_or_create_stock("AAPL")
    assert stock.ticker == "AAPL"
    assert stock.id is not None
    
    # Should return existing stock
    stock2 = data_fetcher._get_or_create_stock("AAPL")
    assert stock2.id == stock.id


def test_store_data(data_fetcher, sample_stock, sample_price_data):
    """Test storing price data in database."""
    data_fetcher._store_data(sample_stock.id, "1d", sample_price_data)
    
    # Check data was stored
    stored_data = data_fetcher.db.query(PriceData).filter(
        PriceData.stock_id == sample_stock.id
    ).all()
    
    assert len(stored_data) == len(sample_price_data)


def test_get_data_from_db(data_fetcher, sample_stock, sample_price_data):
    """Test retrieving data from database."""
    # Store data first
    data_fetcher._store_data(sample_stock.id, "1d", sample_price_data)
    
    # Retrieve data
    df = data_fetcher.get_data_from_db("TEST", "1d", limit=100)
    
    assert len(df) == len(sample_price_data)
    assert 'close' in df.columns
    assert 'timestamp' in df.columns


def test_map_timeframe_to_interval(data_fetcher):
    """Test timeframe mapping."""
    assert data_fetcher._map_timeframe_to_interval("1d") == "1d"
    assert data_fetcher._map_timeframe_to_interval("1h") == "1h"
    assert data_fetcher._map_timeframe_to_interval("5m") == "5m"

