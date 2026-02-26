import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy import and_

from app.models.database import Stock, PriceData
from config import settings

logger = logging.getLogger(__name__)

# Market timezone: PST (America/Los_Angeles) for "today" so we never query future dates
PST = ZoneInfo("America/Los_Angeles")


def _max_query_end_utc() -> datetime:
    """Latest allowed end for yfinance requests: start of tomorrow PST as naive UTC. Never query the future."""
    now_pst = datetime.now(PST)
    tomorrow_start_pst = now_pst.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
    return tomorrow_start_pst.astimezone(timezone.utc).replace(tzinfo=None)


def _max_query_end_for_timeframe(timeframe: str) -> datetime:
    """For intraday, cap end to 'now' so we never request future minutes (avoids Yahoo 'data doesn't exist')."""
    end = _max_query_end_utc()
    if timeframe in ("1m", "5m", "15m", "1h"):
        now_utc = datetime.utcnow()
        # Cap to now + 2 minutes so the latest incomplete candle can still be returned
        end = min(end, now_utc + timedelta(minutes=2))
    return end


# Yahoo Finance uses BTC-USD, ETH-USD etc. for crypto; user may pass BTC, ETH
def _yfinance_ticker(ticker: str) -> str:
    """Return the ticker symbol to use when calling yfinance (e.g. BTC -> BTC-USD)."""
    u = ticker.upper()
    if u == "BTC":
        return "BTC-USD"
    if u == "ETH":
        return "ETH-USD"
    return ticker


def _timestamp_to_naive_utc(ts) -> datetime:
    """Normalize a timestamp to naive UTC for consistent DB storage and comparison."""
    if ts is None:
        return None
    if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
        from datetime import timezone
        return ts.astimezone(timezone.utc).replace(tzinfo=None)
    return ts


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
        yf_symbol = _yfinance_ticker(ticker)
        yf_ticker = yf.Ticker(yf_symbol)
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
            # Normalize to naive UTC so DB store/check matches (include current date in DB)
            df['timestamp'] = df['timestamp'].apply(
                lambda x: _timestamp_to_naive_utc(x) if pd.notna(x) else x
            )
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
        # Cap at max candles per (stock, timeframe); DB stores and returns at most this many
        max_candles = getattr(settings, 'CACHE_MAX_CANDLES', 1000)
        candles = min(candles, max_candles)
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
            # For intraday, cap end to "now" so we don't request future minutes (avoids Yahoo "data doesn't exist")
            end_dt = _max_query_end_for_timeframe(timeframe)
            if start_dt >= end_dt:
                return self._get_cached_data(stock.id, timeframe, candles)
            # Don't request if start is already "tomorrow" in PST (would query future)
            today_pst = datetime.now(PST).date()
            start_date_pst = (start_dt.replace(tzinfo=timezone.utc) if start_dt.tzinfo is None else start_dt).astimezone(PST).date()
            if start_date_pst > today_pst:
                return self._get_cached_data(stock.id, timeframe, candles)
            self._fetch_incremental(stock.id, ticker, timeframe, start_dt, end_dt)
            # Return most recent `candles` from DB for this timeframe (may be fewer if not enough data)
            return self._get_cached_data(stock.id, timeframe, candles)
        
        # No cache or not enough candles -> full fetch
        logger.info(f"Fetching {candles} candles of {timeframe} data for {ticker}")
        
        # Yahoo Finance limits intraday (1h, 15m, 5m, 1m) to the last 60 days (7 for 1m).
        # So we may get fewer bars than requested for intraday timeframes.
        intraday = timeframe in ('1m', '5m', '15m', '1h')
        
        max_retries = 3
        df = pd.DataFrame()  # Initialize empty DataFrame
        
        yf_symbol = _yfinance_ticker(ticker)
        for attempt in range(max_retries):
            try:
                yf_ticker = yf.Ticker(yf_symbol)
                interval = self._map_timeframe_to_interval(timeframe)
                # For intraday, cap end to "now" so we don't request future minutes (avoids Yahoo "data doesn't exist")
                end_date = _max_query_end_for_timeframe(timeframe)
                
                # For intraday, use explicit start/end (Yahoo recommends this) and cap at 60 days (7 for 1m)
                if intraday:
                    if timeframe == '1m':
                        days_back = 7   # Yahoo caps 1m at ~7 days
                    else:
                        days_back = 60  # Yahoo caps other intraday at 60 days
                    start_date = end_date - timedelta(days=days_back)
                    logger.info(f"Attempt {attempt + 1}: Using date range {start_date} to {end_date} (intraday max {days_back}d, ticker={yf_symbol})")
                    df = yf_ticker.history(start=start_date, end=end_date, interval=interval)
                elif attempt == 0:
                    # Daily: use start/end with end capped to today PST
                    end_date = _max_query_end_utc()
                    days_back = max(365, int(candles * 1.5)) if timeframe == '1d' else 60
                    start_date = end_date - timedelta(days=days_back)
                    logger.info(f"Attempt {attempt + 1}: Using date range {start_date.date()} to {end_date.date()} (ticker={yf_symbol})")
                    df = yf_ticker.history(start=start_date, end=end_date, interval=interval)
                else:
                    # Fallback: use date range for daily (end capped to today PST)
                    end_date = _max_query_end_utc()
                    days_back = max(365, int(candles * 1.5))
                    start_date = end_date - timedelta(days=days_back)
                    logger.info(f"Attempt {attempt + 1}: Using date range {start_date.date()} to {end_date.date()} (ticker={yf_symbol})")
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
        
        # Ensure timestamp is datetime and normalize to naive UTC for consistent DB storage
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['timestamp'] = df['timestamp'].apply(
                lambda x: _timestamp_to_naive_utc(x) if pd.notna(x) else x
            )
        
        # Select only required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Limit to requested number of candles (may have fewer if provider limits apply)
        df = df.tail(candles).reset_index(drop=True)
        
        if len(df) < candles:
            logger.warning(
                f"Requested {candles} candles but received {len(df)}. "
                f"For intraday timeframes (1h, 15m, 5m, 1m), Yahoo Finance limits data to the last 60 days (7 for 1m), "
                f"so fewer bars than requested may be returned and stored."
            )
        
        self._store_data(stock.id, timeframe, df)

        # Trim only when past CACHE_MAX_CANDLES for this (stock, timeframe)
        self._trim_cache(stock.id, timeframe)
        
        logger.info(f"Successfully fetched and stored {len(df)} candles for {ticker}")
        
        return df
    
    # Stores price data in database (timestamps should be naive UTC so current date is not skipped)
    def _store_data(self, stock_id: int, timeframe: str, df: pd.DataFrame):
        try:
            for _, row in df.iterrows():
                ts = row['timestamp']
                if ts is not None and (hasattr(ts, "tzinfo") and ts.tzinfo is not None):
                    ts = _timestamp_to_naive_utc(ts)
                # Check if record already exists (compare normalized timestamp)
                existing = (
                    self.db.query(PriceData)
                    .filter(
                        and_(
                            PriceData.stock_id == stock_id,
                            PriceData.timeframe == timeframe,
                            PriceData.timestamp == ts
                        )
                    )
                    .first()
                )
                
                if not existing:
                    price_data = PriceData(
                        stock_id=stock_id,
                        timestamp=ts,
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
        # Gets data directly from database without fetching from yfinance. Returns at most CACHE_MAX_CANDLES.
        max_candles = getattr(settings, 'CACHE_MAX_CANDLES', 1000)
        limit = min(limit, max_candles)
        stock = self.db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
        if not stock:
            raise ValueError(f"Stock {ticker} not found in database")
        # Return the most recent `limit` rows in chronological order
        data = (
            self.db.query(PriceData)
            .filter(
                and_(
                    PriceData.stock_id == stock.id,
                    PriceData.timeframe == timeframe
                )
            )
            .order_by(PriceData.timestamp.desc())
            .limit(limit)
            .all()
        )
        
        if not data:
            raise ValueError(f"No data found for {ticker} with timeframe {timeframe}")
        # Reverse so result is chronological (oldest first)
        data = list(reversed(data))
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

