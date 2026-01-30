"""Manual database initialization script."""
import logging
from app.database import init_db, engine
from app.models.database import Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Initialize the database manually."""
    logger.info("Initializing database...")
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✓ Database initialized successfully!")
        logger.info("Tables created: stocks, price_data, strategies, backtests")
    except Exception as e:
        logger.error(f"✗ Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    main()






