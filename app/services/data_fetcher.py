import logging
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.database import Stock, PriceData
from config import settings

logger = logging.getLogger(__name__)


# Service for fetching and caching stock price data using yfinance
class DataFetcher:
    def __init__(self, db: Session):
        self.db = db
    
    # Gets existing stock or creates new one in database
    def _get_or_create_stock(self, ticker: str) -> Stock:
        stock = self.db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
        if not stock:
            stock = Stock(ticker=ticker.upper())
            self.db.add(stock)
            self.db.commit()
            self.db.refresh(stock)
        return stock
    
    # Checks if we have enough cached data in database
    def _check_cache(self, stock_id: int, timeframe: str, required_candles: int) -> Optional[pd.DataFrame]:
        if not settings.CACHE_ENABLED:
            return None
        
        # Get the most recent data
        latest_data = (
            self.db.query(PriceData)
            .filter(
                and_(
                    PriceData.stock_id == stock_id,
                    PriceData.timeframe == timeframe
                )
            )
            .order_by(PriceData.timestamp.desc())
            .limit(required_candles)
            .all()
        )
        
        if len(latest_data) >= required_candles:
            # Convert to DataFrame
            data = []
            for row in reversed(latest_data):  # Reverse to get chronological order
                data.append({
                    'timestamp': row.timestamp,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume
                })
            df = pd.DataFrame(data)
            logger.info(f"Using cached data: {len(df)} candles for {timeframe}")
            return df
        
        return None
    
    # Returns (count, latest_timestamp) for cached data for this stock+timeframe; (0, None) if none
    def _get_cache_info(self, stock_id: int, timeframe: str) -> tuple:
        if not settings.CACHE_ENABLED:
            return 0, None
        count = (
            self.db.query(PriceData)
            .filter(
                and_(
                    PriceData.stock_id == stock_id,
                    PriceData.timeframe == timeframe
                )
            )
            .count()
        )
        if count == 0:
            return 0, None
        latest = (
            self.db.query(PriceData.timestamp)
            .filter(
                and_(
                    PriceData.stock_id == stock_id,
                    PriceData.timeframe == timeframe
                )
            )
            .order_by(PriceData.timestamp.desc())
            .limit(1)
            .first()
        )
        return count, latest[0] if latest else None
    
    # True if cache is behind "now" for this timeframe (so we should fetch missing candles)
    def _is_cache_stale(self, latest_ts, timeframe: str) -> bool:
        if latest_ts is None:
            return True
        now = datetime.utcnow()
        # Normalize: make latest_ts comparable with naive UTC now
        if hasattr(latest_ts, 'tzinfo') and latest_ts.tzinfo is not None:
            from datetime import timezone
            latest_ts = latest_ts.astimezone(timezone.utc).replace(tzinfo=None)
        if timeframe == '1d':
            latest_date = latest_ts.date() if hasattr(latest_ts, 'date') else latest_ts
            return latest_date < now.date()
        if timeframe == '1h':
            return (now - latest_ts).total_seconds() > 3600
        if timeframe == '15m':
            return (now - latest_ts).total_seconds() > 15 * 60
        if timeframe == '5m':
            return (now - latest_ts).total_seconds() > 5 * 60
        if timeframe == '1m':
            return (now - latest_ts).total_seconds() > 60
        return True
    
    # Start of next period after latest_ts for this timeframe (for incremental fetch)
    def _next_period_start(self, latest_ts, timeframe: str):
        if hasattr(latest_ts, 'tzinfo') and latest_ts.tzinfo:
            # Keep timezone for yfinance
            pass
        if timeframe == '1d':
            return latest_ts + timedelta(days=1)
        if timeframe == '1h':
            return latest_ts + timedelta(hours=1)
        if timeframe == '15m':
            return latest_ts + timedelta(minutes=15)
        if timeframe == '5m':
            return latest_ts + timedelta(minutes=5)
        if timeframe == '1m':
            return latest_ts + timedelta(minutes=1)
        return latest_ts + timedelta(days=1)
    
    # Trims cache for this (stock_id, timeframe) to at most CACHE_MAX_CANDLES; only deletes when past that mark
    def _trim_cache(self, stock_id: int, timeframe: str):
        max_candles = getattr(settings, 'CACHE_MAX_CANDLES', 1000)
        count = (
            self.db.query(PriceData)
            .filter(
                and_(
                    PriceData.stock_id == stock_id,
                    PriceData.timeframe == timeframe
                )
            )
            .count()
        )
        if count <= max_candles:
            return
        # Get timestamp of the max_candles-th newest row (so we keep that and newer)
        thousandth_row = (
            self.db.query(PriceData)
            .filter(
                and_(
                    PriceData.stock_id == stock_id,
                    PriceData.timeframe == timeframe
                )
            )
            .order_by(PriceData.timestamp.desc())
            .offset(max_candles - 1)
            .limit(1)
            .first()
        )
        if not thousandth_row:
            return
        cutoff = thousandth_row.timestamp
        deleted = (
            self.db.query(PriceData)
            .filter(
                and_(
                    PriceData.stock_id == stock_id,
                    PriceData.timeframe == timeframe,
                    PriceData.timestamp < cutoff
                )
            )
            .delete(synchronize_session=False)
        )
        self.db.commit()
        logger.info(f"Trimmed cache for stock_id={stock_id} timeframe={timeframe}: removed {deleted} rows (kept {max_candles})")
    
    # Returns up to max_candles most recent rows for (stock_id, timeframe) in chronological order
    def _get_cached_data(self, stock_id: int, timeframe: str, max_candles: int) -> pd.DataFrame:
        rows = (
            self.db.query(PriceData)
            .filter(
                and_(
                    PriceData.stock_id == stock_id,
                    PriceData.timeframe == timeframe
                )
            )
            .order_by(PriceData.timestamp.desc())
            .limit(max_candles)
            .all()
        )
        if not rows:
            return pd.DataFrame()
        data = [
            {'timestamp': r.timestamp, 'open': r.open, 'high': r.high, 'low': r.low, 'close': r.close, 'volume': r.volume}
            for r in reversed(rows)
        ]
        return pd.DataFrame(data)
    
    # Fetches only missing candles from start_dt to end_dt for this timeframe; appends to DB, then trims if > CACHE_MAX_CANDLES
    def _fetch_incremental(
        self,
        stock_id: int,
        ticker: str,
        timeframe: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> pd.DataFrame:
        interval = self._map_timeframe_to_interval(timeframe)
        yf_ticker = yf.Ticker(ticker)
        df = yf_ticker.history(start=start_dt, end=end_dt, interval=interval)
        if df.empty:
            logger.info(f"Incremental fetch for {ticker} {timeframe}: no new data from {start_dt} to {end_dt}")
            return pd.DataFrame()
        df.reset_index(inplace=True)
        df.rename(columns={
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        self._store_data(stock_id, timeframe, df)
        self._trim_cache(stock_id, timeframe)
        logger.info(f"Incremental fetch for {ticker} {timeframe}: stored {len(df)} new candles")
        return df
    
    # Maps our timeframe format to yfinance interval format
    def _map_timeframe_to_interval(self, timeframe: str) -> str:
        mapping = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '1h': '1h',
            '1d': '1d'
        }
        return mapping.get(timeframe, '1d')
    
    # Calculates yfinance period parameter based on timeframe and candles
    def _calculate_period(self, timeframe: str, candles: int) -> str:
        # yfinance period format: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        # Add buffer to account for weekends/holidays
        if timeframe == '1m':
            days = max(7, candles // (60 * 24))  # At least 7 days
            return f"{days}d" if days < 60 else "2mo"
        elif timeframe == '5m':
            days = max(7, candles // (12 * 24))
            return f"{days}d" if days < 60 else "2mo"
        elif timeframe == '15m':
            days = max(7, candles // (4 * 24))
            return f"{days}d" if days < 60 else "2mo"
        elif timeframe == '1h':
            days = max(7, candles // 24)
            return f"{days}d" if days < 60 else "2mo"
        else:  # 1d
            # For daily data, add buffer for weekends/holidays (multiply by ~1.4)
            # and ensure minimum period
            days = max(30, int(candles * 1.4))  # At least 1 month, buffer for weekends
            if days < 60:
                return "1mo"  # Use 1mo minimum for daily data
            elif days < 90:
                return "3mo"
            elif days < 180:
                return "6mo"
            elif days < 365:
                return "1y"
            elif days < 730:
                return "2y"
            else:
                return "5y"
    
    def fetch_data(
        self,
        ticker: str,
        timeframe: str = "1d",
        candles: int = 1000
    ) -> pd.DataFrame:
        # Fetches historical price data from yfinance or cache; updates cache with missing timestamps only; trims only when > CACHE_MAX_CANDLES
        ticker = ticker.upper()
        stock = self._get_or_create_stock(ticker)
        
        # Cache is per (stock, timeframe): 1000 candles of 5m is only for 5m, not for 1d/1h
        count, latest_ts = self._get_cache_info(stock.id, timeframe)
        
        # Have enough candles and cache is up to date -> use cache
        if count >= candles and not self._is_cache_stale(latest_ts, timeframe):
            cached_data = self._check_cache(stock.id, timeframe, candles)
            if cached_data is not None:
                return cached_data.head(candles)
        
        # Have some cache but behind -> fetch only missing timestamps and append
        if count >= 1 and latest_ts is not None and self._is_cache_stale(latest_ts, timeframe):
            start_dt = self._next_period_start(latest_ts, timeframe)
            end_dt = datetime.utcnow()
            self._fetch_incremental(stock.id, ticker, timeframe, start_dt, end_dt)
            # Return most recent `candles` from DB for this timeframe (may be fewer if not enough data)
            return self._get_cached_data(stock.id, timeframe, candles)
        
        # No cache or not enough candles -> full fetch
        logger.info(f"Fetching {candles} candles of {timeframe} data for {ticker}")
        
        # Try multiple approaches if the first one fails
        max_retries = 3
        df = pd.DataFrame()  # Initialize empty DataFrame
        
        for attempt in range(max_retries):
            try:
                yf_ticker = yf.Ticker(ticker)
                interval = self._map_timeframe_to_interval(timeframe)
                
                # Try period-based fetch first
                if attempt == 0:
                    period = self._calculate_period(timeframe, candles)
                    logger.info(f"Attempt {attempt + 1}: Using period={period}, interval={interval}")
                    df = yf_ticker.history(period=period, interval=interval)
                else:
                    # Fallback: use date range for more reliable fetching
                    end_date = datetime.now()
                    # Calculate start date with buffer
                    if timeframe == '1d':
                        days_back = max(365, int(candles * 1.5))  # At least 1 year, buffer for weekends
                    elif timeframe == '1h':
                        days_back = max(60, int(candles / 24 * 1.2))
                    elif timeframe in ['15m', '5m']:
                        days_back = max(30, int(candles / (4 if timeframe == '15m' else 12) / 24 * 1.2))
                    else:  # 1m
                        days_back = max(7, int(candles / (60 * 24) * 1.2))
                    
                    start_date = end_date - timedelta(days=days_back)
                    logger.info(f"Attempt {attempt + 1}: Using date range {start_date.date()} to {end_date.date()}")
                    df = yf_ticker.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    if attempt < max_retries - 1:
                        logger.warning(f"Empty data on attempt {attempt + 1}, retrying...")
                        continue
                    raise ValueError(f"No data returned for {ticker} after {max_retries} attempts")
                
                # Success - break out of retry loop
                break
                
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"Error on attempt {attempt + 1}: {str(e)}, retrying...")
                    time.sleep(1)  # Brief delay before retry
                else:
                    logger.error(f"All {max_retries} attempts failed for {ticker}")
                    raise ValueError(f"Failed to fetch data for {ticker} after {max_retries} attempts: {str(e)}")
        
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        
        # Reset index to get Date as column
        df.reset_index(inplace=True)
        
        # Rename columns to match our schema (intraday uses 'Datetime', daily uses 'Date')
        df.rename(columns={
            'Date': 'timestamp',
            'Datetime': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }, inplace=True)
        
        # Ensure timestamp is datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Select only required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Limit to requested number of candles
        df = df.tail(candles).reset_index(drop=True)
        
        # Store in database
        self._store_data(stock.id, timeframe, df)
        # Trim only when past CACHE_MAX_CANDLES for this (stock, timeframe)
        self._trim_cache(stock.id, timeframe)
        
        logger.info(f"Successfully fetched and stored {len(df)} candles for {ticker}")
        return df
    
    # Stores price data in database
    def _store_data(self, stock_id: int, timeframe: str, df: pd.DataFrame):
        try:
            for _, row in df.iterrows():
                # Check if record already exists
                existing = (
                    self.db.query(PriceData)
                    .filter(
                        and_(
                            PriceData.stock_id == stock_id,
                            PriceData.timeframe == timeframe,
                            PriceData.timestamp == row['timestamp']
                        )
                    )
                    .first()
                )
                
                if not existing:
                    price_data = PriceData(
                        stock_id=stock_id,
                        timestamp=row['timestamp'],
                        timeframe=timeframe,
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=int(row['volume'])
                    )
                    self.db.add(price_data)
            
            self.db.commit()
            logger.info(f"Stored {len(df)} records in database")
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error storing data: {str(e)}")
            raise
    
    def get_data_from_db(
        self,
        ticker: str,
        timeframe: str = "1d",
        limit: int = 1000
    ) -> pd.DataFrame:
        # Gets data directly from database without fetching from yfinance
        stock = self.db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
        if not stock:
            raise ValueError(f"Stock {ticker} not found in database")
        
        data = (
            self.db.query(PriceData)
            .filter(
                and_(
                    PriceData.stock_id == stock.id,
                    PriceData.timeframe == timeframe
                )
            )
            .order_by(PriceData.timestamp.asc())
            .limit(limit)
            .all()
        )
        
        if not data:
            raise ValueError(f"No data found for {ticker} with timeframe {timeframe}")
        
        df_data = []
        for row in data:
            df_data.append({
                'timestamp': row.timestamp,
                'open': row.open,
                'high': row.high,
                'low': row.low,
                'close': row.close,
                'volume': row.volume
            })
        
        return pd.DataFrame(df_data)

