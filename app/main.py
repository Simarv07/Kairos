import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import init_db
from app.api.routes import router
from app.api.whatsapp_routes import router as whatsapp_router
from app.api.live_routes import router as live_router
from app.services.whatsapp_repeater import whatsapp_repeater
from app.services.live_poller import live_poller
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Kairos Trading Backtest API - stock trading strategy backtesting",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(router)
app.include_router(whatsapp_router)
app.include_router(live_router)


# Initializes database on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    init_db()
    logger.info("Database initialized")


@app.on_event("shutdown")
async def shutdown_event():
    # Stop background tasks cleanly.
    try:
        await whatsapp_repeater.stop()
    except Exception:
        logger.exception("Failed stopping WhatsApp repeater on shutdown")
    try:
        for entry in await live_poller.status():
            await live_poller.stop(entry["poll_id"])
    except Exception:
        logger.exception("Failed stopping live pollers on shutdown")


# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to Kairos Trading Backtest API",
        "version": settings.API_VERSION,
        "docs": "/docs"
    }

