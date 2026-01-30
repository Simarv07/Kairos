# Project Kairos – Trading Backtest API

Kairos is a Python-based backtesting platform with a FastAPI REST interface for evaluating algorithmic trading strategies on historical market data. It supports ARIMA forecasting, Moving Average crossovers, and LSTM models, each configurable with custom parameters. The engine computes key performance metrics such as returns, Sharpe ratio, win rate, drawdown, and trade statistics.

Historical data is fetched through yfinance with automatic caching, and all results are stored using SQLite or PostgreSQL via SQLAlchemy. Backtest runs are fully logged for comparison across tickers, timeframes, and strategies. The platform uses Python 3.9+, FastAPI, SQLAlchemy, yfinance, statsmodels, pandas, numpy, scikit-learn, and pytest.

## Features

The platform includes data fetching with automatic caching, multiple trading strategies (ARIMA, Moving Average, LSTM), comprehensive backtesting with performance metrics, a REST API for triggering backtests and retrieving results, algorithm comparison capabilities, and database support for storing historical data and results.

## Technology Stack

The project is built using Python 3.9+ with FastAPI for the web framework, SQLAlchemy for database operations, yfinance for stock data fetching, statsmodels for ARIMA implementation, pandas and numpy for data manipulation, scikit-learn for metrics calculation, and pytest for testing.

## Setup

`pip install -r requirements.txt`
`python run.py`

The API will be available at http://localhost:8000 with interactive docs at http://localhost:8000/docs.

## API Endpoints

### Fetch Historical Data

POST `/api/fetch-data` - Fetches and caches historical stock price data for a given ticker, timeframe, and number of candles. Returns success status and data fetch information.

### Run Backtest

POST `/api/backtest` - Executes a backtest with a specified strategy, ticker, timeframe, and parameters. Returns comprehensive performance metrics including total return, win rate, Sharpe ratio, and trade details.

### Compare Algorithms

POST `/api/compare-algo` - Compares two algorithms across multiple combinations of tickers, timeframes, and candle counts. Returns detailed per-combination results and an overall comparison summary showing which algorithm performed better.

### Get Backtest Results

GET `/api/results/{backtest_id}` - Retrieves detailed backtest results by ID including all metrics, trades, and model parameters.

### Health Check

GET `/api/health` - Returns API health status and version information.

## Examples

Fetch data for a stock, run a backtest, and retrieve results:

```bash
curl -X POST "http://localhost:8000/api/fetch-data" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "timeframe": "1d", "candles": 1000}'

curl -X POST "http://localhost:8000/api/backtest" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "strategy": "ARIMA", "timeframe": "1d", "candles": 1000, "initial_capital": 10000.0}'

curl "http://localhost:8000/api/results/1"
```

## Testing

Run the test suite with `pytest`

## Project Structure

The project follows a modular structure with separate directories for models (database schemas and Pydantic models), strategies (trading algorithm implementations), services (data fetching, processing, backtesting, and metrics), and API routes (FastAPI endpoints).

## Configuration

Configuration is managed in `config.py` with support for environment variable overrides via `.env` file. Key settings include database URL, default candles, timeframe, ARIMA order, train/test split ratio, initial capital, and transaction costs.

## Performance Metrics

The backtesting engine calculates total return percentage, win rate, maximum drawdown, number of trades, average gain per trade, Sharpe ratio for risk-adjusted returns, and prediction accuracy metrics (MAE, RMSE, MAPE) for model evaluation.

## Strategies

The platform supports ARIMA models with automatic parameter selection, Moving Average crossover strategies with configurable windows, and LSTM neural networks for price prediction. Each strategy can be customized with specific parameters and tested across different market conditions. More to be added later.

## Database

The system uses SQLAlchemy ORM with support for SQLite (default) and PostgreSQL. The database schema includes tables for stocks, price_data, strategies, and backtests, automatically initialized on first run.
