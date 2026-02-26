"""API routes for live price polling: start, stop, status."""
import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import (
    LivePollRequest,
    LivePollResponse,
    LivePollStatusResponse,
    LivePollStatusEntry,
    LivePollStopRequest,
)
from app.services.live_poller import live_poller, LivePollConfig, POLL_INTERVAL_SECONDS
from app.services.whatsapp import TwilioWhatsAppClient
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{settings.API_PREFIX}/live", tags=["live"])


def _get_twilio_client() -> Optional[TwilioWhatsAppClient]:
    """Return Twilio WhatsApp client if configured, else None."""
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN or not settings.TWILIO_WHATSAPP_FROM:
        return None
    return TwilioWhatsAppClient(
        account_sid=settings.TWILIO_ACCOUNT_SID,
        auth_token=settings.TWILIO_AUTH_TOKEN,
        whatsapp_from=settings.TWILIO_WHATSAPP_FROM,
    )


async def _send_whatsapp(to: str, body: str) -> None:
    """Send a WhatsApp message; log and ignore if Twilio not configured or send fails."""
    client = _get_twilio_client()
    if not client:
        logger.warning("WhatsApp not configured; skipping message to %s", to)
        return
    try:
        await client.send_message(to=to, body=body)
    except Exception as e:
        logger.exception("Failed to send WhatsApp message: %s", e)


def _format_live_start_message(ticker: str, strategy: str) -> str:
    """Format the WhatsApp message when live monitoring starts."""
    return (
        f"Monitoring begun for {ticker} for {strategy} strategy. "
        "You may stop monitoring through the app."
    )


@router.post("/start", response_model=LivePollResponse)
async def start_live_poll(request: LivePollRequest):
    """
    Start live price polling for the given ticker and timeframe.
    Data is fetched at an interval based on the timeframe (e.g. 1m → every 60s).
    Keeps a rolling window of 1000 candles; on each new candle the strategy is
    evaluated. If a BUY or SELL signal is triggered, you are notified via WhatsApp
    at the number provided in `whatsapp_to`.
    """
    poll_id = str(uuid.uuid4())
    option_type = (request.option_type or "BOTH").upper()
    if option_type not in ("CALL", "PUT", "BOTH"):
        option_type = "BOTH"
    config = LivePollConfig(
        poll_id=poll_id,
        ticker=request.ticker.upper(),
        timeframe=request.timeframe,
        strategy_name=request.strategy,
        whatsapp_to=request.whatsapp_to.strip(),
        threshold=request.threshold,
        short_window=request.short_window,
        long_window=request.long_window,
        fast_period=request.fast_period,
        slow_period=request.slow_period,
        signal_period=request.signal_period,
        rsi_period=request.rsi_period,
        oversold=request.oversold,
        overbought=request.overbought,
        lookback=request.lookback,
        entry_z=request.entry_z,
        use_options=request.use_options,
        option_type=option_type,
        leap_expiration_days=request.leap_expiration_days or 730,
        strike_selection=(request.strike_selection or "ATM").upper(),
        risk_free_rate=request.risk_free_rate or 0.05,
    )
    try:
        await live_poller.start(config)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    interval = POLL_INTERVAL_SECONDS.get(request.timeframe, 3600)
    start_message = _format_live_start_message(config.ticker, config.strategy_name)
    await _send_whatsapp(request.whatsapp_to.strip(), start_message)

    return LivePollResponse(
        success=True,
        poll_id=poll_id,
        message="Live polling started. You will receive WhatsApp notifications when the strategy triggers a BUY or SELL.",
        ticker=config.ticker,
        timeframe=config.timeframe,
        strategy=config.strategy_name,
        poll_interval_seconds=interval,
    )


@router.post("/stop")
async def stop_live_poll(request: LivePollStopRequest):
    """Stop an active live poll by its poll_id (returned from POST /live/start)."""
    config = await live_poller.stop(request.poll_id)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No active poll with id '{request.poll_id}'",
        )
    stop_message = f"Monitoring stopped for {config.ticker} using {config.strategy_name}"
    await _send_whatsapp(config.whatsapp_to, stop_message)
    return {"success": True, "message": "Poll stopped", "poll_id": request.poll_id}


@router.get("/status", response_model=LivePollStatusResponse)
async def live_poll_status():
    """List all active live poll jobs and their current state."""
    entries = await live_poller.status()
    return LivePollStatusResponse(
        polls=[LivePollStatusEntry(**e) for e in entries],
        message=f"{len(entries)} active poll(s)" if entries else "No active polls",
    )
