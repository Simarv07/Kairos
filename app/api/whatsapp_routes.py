import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.services.whatsapp import TwilioWhatsAppClient
from app.services.whatsapp_repeater import whatsapp_repeater
from config import settings


logger = logging.getLogger(__name__)

router = APIRouter(prefix=f"{settings.API_PREFIX}/whatsapp", tags=["whatsapp"])


def _get_twilio_client() -> TwilioWhatsAppClient:
    if not settings.TWILIO_ACCOUNT_SID or not settings.TWILIO_AUTH_TOKEN or not settings.TWILIO_WHATSAPP_FROM:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Twilio WhatsApp is not configured. Set env vars: "
                "TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM."
            ),
        )
    return TwilioWhatsAppClient(
        account_sid=settings.TWILIO_ACCOUNT_SID,
        auth_token=settings.TWILIO_AUTH_TOKEN,
        whatsapp_from=settings.TWILIO_WHATSAPP_FROM,
    )


class SendOnceRequest(BaseModel):
    to: str = Field(..., description='Recipient WhatsApp address, e.g. "whatsapp:+15551234567" or "+15551234567"')
    message: str = Field(..., description="Message body")


class StartHiRequest(BaseModel):
    to: Optional[str] = Field(
        default=None,
        description='Recipient WhatsApp address. If omitted, uses WHATSAPP_DEFAULT_TO.',
    )
    interval_seconds: int = Field(default=60, ge=5, description="How often to send (seconds)")


@router.get("/status")
async def whatsapp_status():
    return {
        "running": whatsapp_repeater.is_running(),
        "state": {
            "to": whatsapp_repeater.state.to,
            "message": whatsapp_repeater.state.message,
            "interval_seconds": whatsapp_repeater.state.interval_seconds,
            "started_at": whatsapp_repeater.state.started_at,
            "last_sent_at": whatsapp_repeater.state.last_sent_at,
            "last_error": whatsapp_repeater.state.last_error,
        },
    }


@router.post("/send-once")
async def send_once(req: SendOnceRequest):
    client = _get_twilio_client()
    payload = await client.send_message(to=req.to, body=req.message)
    return {"success": True, "twilio": {"sid": payload.get("sid"), "status": payload.get("status")}}


@router.post("/start-hi")
async def start_sending_hi_every_minute(req: StartHiRequest):
    to = req.to or settings.WHATSAPP_DEFAULT_TO
    if not to:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail='Missing "to". Provide it in the request body or set WHATSAPP_DEFAULT_TO.',
        )

    client = _get_twilio_client()
    try:
        await whatsapp_repeater.start(
            client=client,
            to=to,
            message="hi",
            interval_seconds=req.interval_seconds,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    return {"success": True, "message": 'Started sending "hi" every interval', "to": to, "interval_seconds": req.interval_seconds}


@router.post("/stop")
async def stop_sending():
    await whatsapp_repeater.stop()
    return {"success": True, "message": "Stopped WhatsApp repeater (if it was running)"}

