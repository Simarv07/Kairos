import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from app.services.whatsapp import TwilioWhatsAppClient


logger = logging.getLogger(__name__)


@dataclass
class WhatsAppRepeaterState:
    running: bool = False
    to: Optional[str] = None
    message: Optional[str] = None
    interval_seconds: int = 60
    started_at: Optional[datetime] = None
    last_sent_at: Optional[datetime] = None
    last_error: Optional[str] = None


class WhatsAppRepeater:
    def __init__(self) -> None:
        self._task: Optional[asyncio.Task] = None
        self.state = WhatsAppRepeaterState()

    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(
        self,
        *,
        client: TwilioWhatsAppClient,
        to: str,
        message: str = "hi",
        interval_seconds: int = 60,
    ) -> None:
        if self.is_running():
            raise RuntimeError("Repeater is already running")
        if interval_seconds < 5:
            raise ValueError("interval_seconds must be >= 5")

        self.state.running = True
        self.state.to = to
        self.state.message = message
        self.state.interval_seconds = interval_seconds
        self.state.started_at = datetime.now(timezone.utc)
        self.state.last_sent_at = None
        self.state.last_error = None

        async def _loop() -> None:
            try:
                while True:
                    try:
                        await client.send_message(to=to, body=message)
                        self.state.last_sent_at = datetime.now(timezone.utc)
                        self.state.last_error = None
                    except Exception as e:
                        self.state.last_error = str(e)
                        logger.exception("WhatsApp repeater send failed")

                    await asyncio.sleep(interval_seconds)
            except asyncio.CancelledError:
                raise
            finally:
                self.state.running = False

        self._task = asyncio.create_task(_loop(), name="whatsapp_repeater")

    async def stop(self) -> None:
        if not self.is_running():
            self._task = None
            self.state.running = False
            return

        assert self._task is not None
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self.state.running = False


whatsapp_repeater = WhatsAppRepeater()

