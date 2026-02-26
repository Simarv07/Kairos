import logging
from dataclasses import dataclass

import httpx


logger = logging.getLogger(__name__)


def normalize_whatsapp_address(value: str) -> str:
    """
    Twilio expects WhatsApp addresses in the form: "whatsapp:+15551234567".
    Accepts "+1555..." or "whatsapp:+1555..." and normalizes.
    """
    v = (value or "").strip()
    if not v:
        raise ValueError("WhatsApp address is required")
    if v.startswith("whatsapp:"):
        return v
    if v.startswith("+"):
        return f"whatsapp:{v}"
    raise ValueError('WhatsApp address must start with "+" or "whatsapp:+"')


@dataclass(frozen=True)
class TwilioWhatsAppClient:
    account_sid: str
    auth_token: str
    whatsapp_from: str

    async def send_message(self, *, to: str, body: str) -> dict:
        to_addr = normalize_whatsapp_address(to)
        from_addr = normalize_whatsapp_address(self.whatsapp_from)
        if body is None or str(body).strip() == "":
            raise ValueError("Message body is required")

        url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        data = {"From": from_addr, "To": to_addr, "Body": body}

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(url, data=data, auth=(self.account_sid, self.auth_token))

        # Twilio returns JSON for both success and error
        try:
            payload = resp.json()
        except Exception:
            payload = {"raw": resp.text}

        if resp.status_code >= 400:
            message = payload.get("message") if isinstance(payload, dict) else None
            raise RuntimeError(f"Twilio send failed ({resp.status_code}): {message or payload}")

        # success payload includes "sid", "status", etc.
        return payload

