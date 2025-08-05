from typing import AsyncGenerator, Generator
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from app.config.config import settings

# Создаём асинхронный движок и подключение к БД
async_engine = create_async_engine(
    url=settings.database_url_async,
    echo=settings.db_echo,
    future=True,
    pool_size=settings.db_pool_size,
    max_overflow=settings.db_max_overflow,
    pool_timeout=settings.db_pool_timeout,
)

# Создаём синхронный движок для обратной совместимости
sync_engine = create_engine(
    url=settings.database_url,
    echo=settings.db_echo,
    future=True,
)

# Создаём фабрику асинхронных сессий
async_session = async_sessionmaker(
    bind=async_engine,
    autoflush=False,
    autocommit=False,
    class_=AsyncSession,
    expire_on_commit=False,
)

# Создаем фабрику синхронных сессий
sync_session = sessionmaker(
    bind=sync_engine,
    autoflush=False,
    autocommit=False,
    class_=Session,
    expire_on_commit=False,
)

# Базовый класс для моделей
Base = declarative_base()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency для получения асинхронной сессии базы данных"""
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()


def get_sync_db() -> Generator[Session, None, None]:
    """Dependency для получения синхронной сессии базы данных (для обратной совместимости)"""
    with sync_session() as session:
        try:
            yield session
        finally:
            session.close()


# Алиас для обратной совместимости
get_db = get_sync_db
