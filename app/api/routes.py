import logging
from datetime import datetime
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.database import Stock, Backtest, Strategy
from app.models.schemas import (
    FetchDataRequest,
    FetchDataResponse,
    BacktestRequest,
    BacktestResponse,
    BacktestDetailResponse,
    CompareAlgoRequest,
    CompareAlgoResponse,
    AlgorithmConfig,
    AlgorithmMetrics,
    ComparisonResult,
    OverallComparison
)
from app.services.data_fetcher import DataFetcher
from app.services.data_processor import DataProcessor
from app.services.backtester import Backtester
from app.strategies.moving_average import MovingAverageStrategy
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix=settings.API_PREFIX, tags=["trading"])


# Serializes datetime values to ISO format strings
def _serialize_datetime_value(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime().isoformat()
    return str(value)


# Serializes trade history timestamps to ISO format
def _serialize_trade_history(trades):
    serialized = []
    for trade in trades or []:
        serialized.append({
            **trade,
            'entry_time': _serialize_datetime_value(trade.get('entry_time')),
            'exit_time': _serialize_datetime_value(trade.get('exit_time'))
        })
    return serialized


# Deserializes trade history timestamps from ISO format strings
def _deserialize_trade_history(trades):
    def _deserialize(value):
        if value is None or isinstance(value, datetime):
            return value
        if isinstance(value, str):
            iso_value = value.replace('Z', '+00:00')
            try:
                return datetime.fromisoformat(iso_value)
            except ValueError:
                return datetime.fromisoformat(iso_value.strip('"'))
        return value
    deserialized = []
    for trade in trades or []:
        deserialized.append({
            **trade,
            'entry_time': _deserialize(trade.get('entry_time')),
            'exit_time': _deserialize(trade.get('exit_time'))
        })
    return deserialized


# Fetches historical price data for a ticker with caching support
@router.post("/fetch-data", response_model=FetchDataResponse)
async def fetch_data(
    request: FetchDataRequest,
    db: Session = Depends(get_db)
):
    try:
        fetcher = DataFetcher(db)
        
        # Check if data exists in cache
        cached = False
        try:
            existing_data = fetcher.get_data_from_db(request.ticker, request.timeframe, request.candles)
            if len(existing_data) >= request.candles:
                cached = True
                logger.info(f"Using cached data for {request.ticker}")
        except:
            pass
        
        # Fetch data (will use cache if available)
        df = fetcher.fetch_data(
            ticker=request.ticker,
            timeframe=request.timeframe,
            candles=request.candles
        )
        
        return FetchDataResponse(
            success=True,
            message=f"Successfully fetched {len(df)} candles",
            ticker=request.ticker.upper(),
            timeframe=request.timeframe,
            candles_fetched=len(df),
            cached=cached
        )
        
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch data: {str(e)}"
        )


# Runs a backtest with the specified strategy and returns performance metrics
@router.post("/backtest", response_model=BacktestResponse)
async def run_backtest(
    request: BacktestRequest,
    db: Session = Depends(get_db)
):
    try:
        # Fetch data
        fetcher = DataFetcher(db)
        df = fetcher.fetch_data(
            ticker=request.ticker,
            timeframe=request.timeframe,
            candles=request.candles
        )
        
        if len(df) < 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Insufficient data for backtesting (need at least 50 candles)"
            )
        
        # Process data
        processor = DataProcessor()
        df_processed = processor.preprocess_data(df)
        strategy_name = request.strategy.upper()
        if strategy_name in ("MA", "MOVING_AVERAGE"):
            train_df = df_processed
            test_df = df_processed
        else:
            train_df, test_df = processor.split_train_test(df_processed)
        
        # Get or create strategy
        strategy_db = db.query(Strategy).filter(Strategy.name == request.strategy).first()
        if not strategy_db:
            strategy_db = Strategy(name=request.strategy, description=f"{request.strategy} trading strategy")
            db.add(strategy_db)
            db.commit()
            db.refresh(strategy_db)
        
        # Get or create stock
        stock = db.query(Stock).filter(Stock.ticker == request.ticker.upper()).first()
        if not stock:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Stock {request.ticker} not found. Please fetch data first."
            )
        
        # Create backtest record
        backtest = Backtest(
            stock_id=stock.id,
            strategy_id=strategy_db.id,
            start_date=df_processed.iloc[0]['timestamp'] if 'timestamp' in df_processed.columns else datetime.utcnow(),
            end_date=df_processed.iloc[-1]['timestamp'] if 'timestamp' in df_processed.columns else datetime.utcnow(),
            initial_capital=request.initial_capital,
            timeframe=request.timeframe,
            status="running"
        )
        db.add(backtest)
        db.commit()
        db.refresh(backtest)
        
        try:
            if strategy_name in ("MA", "MOVING_AVERAGE"):
                short_window = request.short_window or 20
                long_window = request.long_window or 50
                strategy = MovingAverageStrategy(
                    short_window=short_window,
                    long_window=long_window
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown strategy: {request.strategy}"
                )
            
            # Train on training data
            training_metrics = strategy.fit(train_df)
            
            # Run backtest on test data
            backtester = Backtester(
                initial_capital=request.initial_capital,
                use_options=request.use_options or False,
                leap_expiration_days=request.leap_expiration_days or 730,
                strike_selection=request.strike_selection or 'ATM',
                risk_free_rate=request.risk_free_rate or 0.05
            )
            results = backtester.run(strategy, test_df, threshold=request.threshold)
            final_value = results.get('final_value', request.initial_capital)
            profit_loss = final_value - request.initial_capital
            trade_details = results.get('trades', [])
            serialized_trades = _serialize_trade_history(trade_details)
            
            # Update backtest record
            backtest.status = "completed"
            backtest.completed_at = datetime.utcnow()
            backtest.total_return = results['metrics']['total_return']
            backtest.win_rate = results['metrics']['win_rate']
            backtest.max_drawdown = results['metrics']['max_drawdown']
            backtest.num_trades = results['metrics']['num_trades']
            backtest.avg_gain_per_trade = results['metrics']['avg_gain_per_trade']
            backtest.sharpe_ratio = results['metrics']['sharpe_ratio']
            metrics_payload = {
                **results['metrics'],
                'final_value': final_value,
                'profit_loss': profit_loss,
                'trades': serialized_trades,
                'training_metrics': training_metrics
            }
            backtest.metrics = metrics_payload
            model_parameters = strategy.get_parameters()
            model_parameters['threshold'] = request.threshold
            backtest.model_parameters = model_parameters
            
            db.commit()
            db.refresh(backtest)
            
            return BacktestResponse(
                backtest_id=backtest.id,
                status=backtest.status,
                ticker=request.ticker.upper(),
                strategy=request.strategy,
                start_date=results['start_date'],
                end_date=results['end_date'],
                final_value=final_value,
                profit_loss=profit_loss,
                trades=trade_details,
                total_return=backtest.total_return,
                win_rate=backtest.win_rate,
                max_drawdown=backtest.max_drawdown,
                num_trades=backtest.num_trades,
                avg_gain_per_trade=backtest.avg_gain_per_trade,
                sharpe_ratio=backtest.sharpe_ratio,
                model_parameters=backtest.model_parameters,
                message="Backtest completed successfully"
            )
            
        except Exception as e:
            logger.error(f"Error during backtest execution: {str(e)}")
            backtest.status = "failed"
            backtest.error_message = str(e)
            backtest.completed_at = datetime.utcnow()
            db.commit()
            
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Backtest failed: {str(e)}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run backtest: {str(e)}"
        )


# Retrieves detailed backtest results by ID
@router.get("/results/{backtest_id}", response_model=BacktestDetailResponse)
async def get_backtest_results(
    backtest_id: int,
    db: Session = Depends(get_db)
):
    backtest = db.query(Backtest).filter(Backtest.id == backtest_id).first()
    
    if not backtest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Backtest {backtest_id} not found"
        )
    
    stock = db.query(Stock).filter(Stock.id == backtest.stock_id).first()
    strategy = db.query(Strategy).filter(Strategy.id == backtest.strategy_id).first()
    metrics = backtest.metrics or {}
    final_value = metrics.get('final_value')
    profit_loss = metrics.get('profit_loss')
    trade_details = _deserialize_trade_history(metrics.get('trades'))
    
    return BacktestDetailResponse(
        backtest_id=backtest.id,
        status=backtest.status,
        ticker=stock.ticker if stock else "UNKNOWN",
        strategy=strategy.name if strategy else "UNKNOWN",
        start_date=backtest.start_date,
        end_date=backtest.end_date,
        final_value=final_value,
        profit_loss=profit_loss,
        trades=trade_details,
        initial_capital=backtest.initial_capital,
        timeframe=backtest.timeframe,
        total_return=backtest.total_return,
        win_rate=backtest.win_rate,
        max_drawdown=backtest.max_drawdown,
        num_trades=backtest.num_trades,
        avg_gain_per_trade=backtest.avg_gain_per_trade,
        sharpe_ratio=backtest.sharpe_ratio,
        model_parameters=backtest.model_parameters,
        error_message=backtest.error_message,
        created_at=backtest.created_at,
        completed_at=backtest.completed_at,
        message="Backtest details retrieved successfully"
    )


# Creates a strategy instance from AlgorithmConfig
def _create_strategy_from_config(config: AlgorithmConfig):
    strategy_name = config.strategy.upper()
    
    if strategy_name in ("MA", "MOVING_AVERAGE"):
        short_window = config.short_window or 20
        long_window = config.long_window or 50
        return MovingAverageStrategy(
            short_window=short_window,
            long_window=long_window
        )
    else:
        raise ValueError(f"Unknown strategy: {config.strategy}")


def _run_backtest_for_comparison(
    strategy,
    train_df,
    test_df,
    initial_capital: float,
    threshold: float,
    use_options: bool,
    leap_expiration_days: int,
    strike_selection: str,
    risk_free_rate: float
) -> Dict[str, Any]:
    # Runs a backtest without saving to database, returning results
    # Train strategy
    strategy.fit(train_df)
    
    # Run backtest
    backtester = Backtester(
        initial_capital=initial_capital,
        use_options=use_options,
        leap_expiration_days=leap_expiration_days,
        strike_selection=strike_selection,
        risk_free_rate=risk_free_rate
    )
    results = backtester.run(strategy, test_df, threshold=threshold)
    
    return results


# Extracts metrics from backtest results into AlgorithmMetrics
def _extract_metrics_from_results(results: Dict[str, Any], initial_capital: float) -> AlgorithmMetrics:
    final_value = results.get('final_value', initial_capital)
    profit_loss = final_value - initial_capital
    metrics = results.get('metrics', {})
    
    return AlgorithmMetrics(
        total_return=metrics.get('total_return', 0.0),
        win_rate=metrics.get('win_rate', 0.0),
        max_drawdown=metrics.get('max_drawdown', 0.0),
        num_trades=metrics.get('num_trades', 0),
        avg_gain_per_trade=metrics.get('avg_gain_per_trade', 0.0),
        sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
        final_value=final_value,
        profit_loss=profit_loss
    )


# Compares two algorithm metrics and returns winner (algorithm1, algorithm2, or tie)
def _compare_algorithms(metrics1: AlgorithmMetrics, metrics2: AlgorithmMetrics) -> str:
    # Primary: total_return
    if metrics1.total_return > metrics2.total_return:
        return "algorithm1"
    elif metrics2.total_return > metrics1.total_return:
        return "algorithm2"
    
    # Tie-breaker 1: sharpe_ratio
    if metrics1.sharpe_ratio > metrics2.sharpe_ratio:
        return "algorithm1"
    elif metrics2.sharpe_ratio > metrics1.sharpe_ratio:
        return "algorithm2"
    
    # Tie-breaker 2: win_rate
    if metrics1.win_rate > metrics2.win_rate:
        return "algorithm1"
    elif metrics2.win_rate > metrics1.win_rate:
        return "algorithm2"
    
    return "tie"


# Compares two algorithms across all combinations of tickers, timeframes, and candles
@router.post("/compare-algo", response_model=CompareAlgoResponse)
async def compare_algorithms(
    request: CompareAlgoRequest,
    db: Session = Depends(get_db)
):
    try:
        results = []
        fetcher = DataFetcher(db)
        processor = DataProcessor()
        
        # Iterate through all combinations
        for ticker in request.tickers:
            for timeframe in request.timeframes:
                for candles_count in request.candles:
                    try:
                        # Fetch data
                        df = fetcher.fetch_data(
                            ticker=ticker.upper(),
                            timeframe=timeframe,
                            candles=candles_count
                        )
                        
                        if len(df) < 50:
                            results.append(ComparisonResult(
                                ticker=ticker.upper(),
                                timeframe=timeframe,
                                candles=candles_count,
                                algorithm1_name=request.algorithm1.strategy,
                                algorithm2_name=request.algorithm2.strategy,
                                algorithm1_metrics=AlgorithmMetrics(
                                    total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                                    num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                                    final_value=request.initial_capital, profit_loss=0.0
                                ),
                                algorithm2_metrics=AlgorithmMetrics(
                                    total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                                    num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                                    final_value=request.initial_capital, profit_loss=0.0
                                ),
                                winner="tie",
                                error=f"Insufficient data: only {len(df)} candles available"
                            ))
                            continue
                        
                        # Process data
                        df_processed = processor.preprocess_data(df)
                        strategy1_name = request.algorithm1.strategy.upper()
                        strategy2_name = request.algorithm2.strategy.upper()
                        
                        if strategy1_name in ("MA", "MOVING_AVERAGE") or strategy2_name in ("MA", "MOVING_AVERAGE"):
                            train_df = df_processed
                            test_df = df_processed
                        else:
                            train_df, test_df = processor.split_train_test(df_processed)
                        
                        # Run algorithm 1
                        try:
                            strategy1 = _create_strategy_from_config(request.algorithm1)
                            results1 = _run_backtest_for_comparison(
                                strategy1,
                                train_df,
                                test_df,
                                request.initial_capital,
                                request.threshold,
                                request.use_options or False,
                                request.leap_expiration_days or 730,
                                request.strike_selection or 'ATM',
                                request.risk_free_rate or 0.05
                            )
                            metrics1 = _extract_metrics_from_results(results1, request.initial_capital)
                        except Exception as e:
                            logger.error(f"Error running algorithm1 for {ticker}/{timeframe}/{candles_count}: {str(e)}")
                            metrics1 = AlgorithmMetrics(
                                total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                                num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                                final_value=request.initial_capital, profit_loss=0.0
                            )
                            results.append(ComparisonResult(
                                ticker=ticker.upper(),
                                timeframe=timeframe,
                                candles=candles_count,
                                algorithm1_name=request.algorithm1.strategy,
                                algorithm2_name=request.algorithm2.strategy,
                                algorithm1_metrics=metrics1,
                                algorithm2_metrics=AlgorithmMetrics(
                                    total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                                    num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                                    final_value=request.initial_capital, profit_loss=0.0
                                ),
                                winner="algorithm2",
                                error=f"Algorithm1 failed: {str(e)}"
                            ))
                            continue
                        
                        # Run algorithm 2
                        try:
                            strategy2 = _create_strategy_from_config(request.algorithm2)
                            results2 = _run_backtest_for_comparison(
                                strategy2,
                                train_df,
                                test_df,
                                request.initial_capital,
                                request.threshold,
                                request.use_options or False,
                                request.leap_expiration_days or 730,
                                request.strike_selection or 'ATM',
                                request.risk_free_rate or 0.05
                            )
                            metrics2 = _extract_metrics_from_results(results2, request.initial_capital)
                        except Exception as e:
                            logger.error(f"Error running algorithm2 for {ticker}/{timeframe}/{candles_count}: {str(e)}")
                            metrics2 = AlgorithmMetrics(
                                total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                                num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                                final_value=request.initial_capital, profit_loss=0.0
                            )
                            results.append(ComparisonResult(
                                ticker=ticker.upper(),
                                timeframe=timeframe,
                                candles=candles_count,
                                algorithm1_name=request.algorithm1.strategy,
                                algorithm2_name=request.algorithm2.strategy,
                                algorithm1_metrics=metrics1,
                                algorithm2_metrics=metrics2,
                                winner="algorithm1",
                                error=f"Algorithm2 failed: {str(e)}"
                            ))
                            continue
                        
                        # Compare results
                        winner = _compare_algorithms(metrics1, metrics2)
                        
                        results.append(ComparisonResult(
                            ticker=ticker.upper(),
                            timeframe=timeframe,
                            candles=candles_count,
                            algorithm1_name=request.algorithm1.strategy,
                            algorithm2_name=request.algorithm2.strategy,
                            algorithm1_metrics=metrics1,
                            algorithm2_metrics=metrics2,
                            winner=winner
                        ))
                        
                    except Exception as e:
                        logger.error(f"Error processing combination {ticker}/{timeframe}/{candles_count}: {str(e)}")
                        results.append(ComparisonResult(
                            ticker=ticker.upper(),
                            timeframe=timeframe,
                            candles=candles_count,
                            algorithm1_name=request.algorithm1.strategy,
                            algorithm2_name=request.algorithm2.strategy,
                            algorithm1_metrics=AlgorithmMetrics(
                                total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                                num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                                final_value=request.initial_capital, profit_loss=0.0
                            ),
                            algorithm2_metrics=AlgorithmMetrics(
                                total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                                num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                                final_value=request.initial_capital, profit_loss=0.0
                            ),
                            winner="tie",
                            error=str(e)
                        ))
        
        # Calculate overall comparison
        successful_results = [r for r in results if r.error is None]
        total_combinations = len(request.tickers) * len(request.timeframes) * len(request.candles)
        successful_combinations = len(successful_results)
        
        algorithm1_wins = sum(1 for r in successful_results if r.winner == "algorithm1")
        algorithm2_wins = sum(1 for r in successful_results if r.winner == "algorithm2")
        ties = sum(1 for r in successful_results if r.winner == "tie")
        
        # Calculate average metrics
        if successful_combinations > 0:
            avg_metrics1 = AlgorithmMetrics(
                total_return=sum(r.algorithm1_metrics.total_return for r in successful_results) / successful_combinations,
                win_rate=sum(r.algorithm1_metrics.win_rate for r in successful_results) / successful_combinations,
                max_drawdown=sum(r.algorithm1_metrics.max_drawdown for r in successful_results) / successful_combinations,
                num_trades=int(sum(r.algorithm1_metrics.num_trades for r in successful_results) / successful_combinations),
                avg_gain_per_trade=sum(r.algorithm1_metrics.avg_gain_per_trade for r in successful_results) / successful_combinations,
                sharpe_ratio=sum(r.algorithm1_metrics.sharpe_ratio for r in successful_results) / successful_combinations,
                final_value=sum(r.algorithm1_metrics.final_value for r in successful_results) / successful_combinations,
                profit_loss=sum(r.algorithm1_metrics.profit_loss for r in successful_results) / successful_combinations
            )
            avg_metrics2 = AlgorithmMetrics(
                total_return=sum(r.algorithm2_metrics.total_return for r in successful_results) / successful_combinations,
                win_rate=sum(r.algorithm2_metrics.win_rate for r in successful_results) / successful_combinations,
                max_drawdown=sum(r.algorithm2_metrics.max_drawdown for r in successful_results) / successful_combinations,
                num_trades=int(sum(r.algorithm2_metrics.num_trades for r in successful_results) / successful_combinations),
                avg_gain_per_trade=sum(r.algorithm2_metrics.avg_gain_per_trade for r in successful_results) / successful_combinations,
                sharpe_ratio=sum(r.algorithm2_metrics.sharpe_ratio for r in successful_results) / successful_combinations,
                final_value=sum(r.algorithm2_metrics.final_value for r in successful_results) / successful_combinations,
                profit_loss=sum(r.algorithm2_metrics.profit_loss for r in successful_results) / successful_combinations
            )
            
            # Determine overall winner
            overall_winner = _compare_algorithms(avg_metrics1, avg_metrics2)
        else:
            avg_metrics1 = AlgorithmMetrics(
                total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                final_value=request.initial_capital, profit_loss=0.0
            )
            avg_metrics2 = AlgorithmMetrics(
                total_return=0.0, win_rate=0.0, max_drawdown=0.0,
                num_trades=0, avg_gain_per_trade=0.0, sharpe_ratio=0.0,
                final_value=request.initial_capital, profit_loss=0.0
            )
            overall_winner = "tie"
        
        overall_comparison = OverallComparison(
            algorithm1_name=request.algorithm1.strategy,
            algorithm2_name=request.algorithm2.strategy,
            total_combinations=total_combinations,
            successful_combinations=successful_combinations,
            algorithm1_wins=algorithm1_wins,
            algorithm2_wins=algorithm2_wins,
            ties=ties,
            algorithm1_avg_metrics=avg_metrics1,
            algorithm2_avg_metrics=avg_metrics2,
            overall_winner=overall_winner
        )
        
        return CompareAlgoResponse(
            results=results,
            overall_comparison=overall_comparison,
            message=f"Comparison completed: {successful_combinations}/{total_combinations} combinations successful"
        )
        
    except Exception as e:
        logger.error(f"Error comparing algorithms: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare algorithms: {str(e)}"
        )


# Health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "healthy", "version": settings.API_VERSION}

