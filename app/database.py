# database.py

import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.engine import URL
from sqlalchemy import text
from . import config

logger = logging.getLogger(__name__)

# ---------- Main engine for the target database ----------
url = URL.create(
    drivername="mysql+aiomysql",
    username=config.MYSQL_USER,
    password=config.MYSQL_PASSWORD,
    host=config.MYSQL_HOST,
    port=config.MYSQL_PORT,
    database=config.MYSQL_DB,
)

engine = create_async_engine(
    url,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=3600,
)

AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    """Ensure the database exists, then create tables."""
    # First, connect to a default database (e.g., 'mysql') to create the target DB
    default_url = URL.create(
        drivername="mysql+aiomysql",
        username=config.MYSQL_USER,
        password=config.MYSQL_PASSWORD,
        host=config.MYSQL_HOST,
        port=config.MYSQL_PORT,
        database="mysql",  # Always exists
    )
    default_engine = create_async_engine(default_url, echo=False, pool_pre_ping=True)

    try:
        async with default_engine.connect() as conn:
            # Create database if not exists
            await conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {config.MYSQL_DB}"))
            await conn.commit()
            logger.info(f"✅ Database '{config.MYSQL_DB}' ensured.")
    except Exception as e:
        logger.error(f"❌ Failed to create database: {e}")
        raise
    finally:
        await default_engine.dispose()

    # Import ORM models before creating tables so metadata is populated.
    # (User, Fabric, Room all live in models.py now.)
    from . import models  # noqa: F401

    # Now connect to the target database and create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("✅ Tables created/verified.")
