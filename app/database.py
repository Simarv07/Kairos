from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager

from app.models.database import Base
from config import settings

# Create database engine
engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {}
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Initializes database tables
def init_db():
    Base.metadata.create_all(bind=engine)


# Gets database session (dependency for FastAPI)
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Gets database session as context manager
@contextmanager
def get_db_context():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

