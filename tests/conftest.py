import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock

import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from app.db.database import get_async_db
from app.main import app
from app.models.models import Base, ChatMessage, ChatSession, Document, User
from app.rag.rag_service import RAGService
from app.schemas.schemas import RAGStatus
from app.utils.auth import create_access_token, get_password_hash

# Тестовая база данных
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Создаем тестовый движок
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    poolclass=StaticPool,
    echo=False,
)

# Создаем тестовую сессию
TestingSessionLocal = async_sessionmaker(
    test_engine, class_=AsyncSession, expire_on_commit=False
)


@pytest_asyncio.fixture(scope="session")
def event_loop() -> Generator:
    """Создание event loop для тестов"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="session", autouse=True)
async def setup_database():
    """Настройка тестовой базы данных"""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Фикстура для тестовой сессии БД"""
    async with TestingSessionLocal() as session:
        async with session.begin():
            yield session


@pytest_asyncio.fixture
def client() -> Generator:
    """Фикстура для тестового клиента"""

    async def override_get_db():
        async with TestingSessionLocal() as session:
            yield session

    app.dependency_overrides[get_async_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="session")
async def test_user() -> User:
    """Создание тестового пользователя"""
    async with TestingSessionLocal() as session:
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=get_password_hash("testpassword"),
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


@pytest_asyncio.fixture(scope="session")
async def test_user2() -> User:
    """Создание второго тестового пользователя"""
    async with TestingSessionLocal() as session:
        user = User(
            username="testuser2",
            email="test2@example.com",
            hashed_password=get_password_hash("testpassword"),
            is_active=True,
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


@pytest_asyncio.fixture
async def auth_headers(test_user: User) -> dict:
    """Заголовки с токеном аутентификации"""
    token = create_access_token(data={"sub": test_user.username})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def auth_headers_user2(test_user2: User) -> dict:
    """Заголовки с токеном второго пользователя"""
    token = create_access_token(data={"sub": test_user2.username})
    return {"Authorization": f"Bearer {token}"}


@pytest_asyncio.fixture
async def test_document(test_user: User) -> Document:
    """Создание тестового документа"""
    async with TestingSessionLocal() as session:
        document = Document(
            title="Тестовый документ",
            content="Это тестовый документ для проверки RAG системы.",
            file_path="test.txt",
            file_type=".txt",
            chunk_size=1000,
            chunk_overlap=200,
            owner_id=test_user.id,
            is_processed=True,
        )
        session.add(document)
        await session.flush()
        await session.refresh(document)
        return document


@pytest_asyncio.fixture
async def test_chat_session(test_user: User) -> AsyncGenerator[ChatSession, None]:
    """Создание тестовой сессии чата с откатом транзакции"""
    async with TestingSessionLocal() as session:
        async with session.begin():  # Начинаем транзакцию
            chat_session = ChatSession(title="Тестовая сессия", user_id=test_user.id)
            session.add(chat_session)
            await session.flush()
            await session.refresh(chat_session)
            yield chat_session


@pytest_asyncio.fixture
async def test_chat_message(test_chat_session: ChatSession) -> ChatMessage:
    """Создание тестового сообщения чата"""
    async with TestingSessionLocal() as session:
        message = ChatMessage(
            content="Тестовое сообщение",
            role="user",
            session_id=test_chat_session.id,
        )
        session.add(message)
        await session.commit()
        await session.refresh(message)
        return message


@pytest_asyncio.fixture
async def mock_rag_service():
    """Мок RAG сервиса"""
    service = MagicMock(spec=RAGService)

    # Mock attributes
    service.embedding_model = MagicMock()
    service.vector_store = AsyncMock()
    service.text_splitter = MagicMock()

    # Mock sync methods
    service.embedding_model.encode.return_value = [[0.1, 0.2], [0.3, 0.4]]
    service.embedding_model.encode_single.return_value = [0.1, 0.2]
    service.text_splitter.split_text.return_value = ["Чанк 1", "Чанк 2"]

    # Mock async vector_store methods
    service.vector_store.add_texts = AsyncMock(return_value=["id1", "id2"])
    service.vector_store.search = AsyncMock(
        return_value=[
            ("Чанк 1", 0.9, {"document_id": 1}),
            ("Чанк 2", 0.8, {"document_id": 2}),
        ]
    )
    service.vector_store.get_collection_info = AsyncMock(
        return_value={
            "points_count": 5,
            "vectors_count": 5,
        }
    )
    service.vector_store.delete_by_document_id = AsyncMock(return_value=True)
    service.vector_store.delete_texts = AsyncMock(return_value=True)
    service.vector_store._ensure_collection_exists = AsyncMock(
        return_value=None
    )  # Mock to prevent Qdrant connection

    # Mock RAGStatus
    default_status = RAGStatus(
        total_documents=1,
        processed_documents=1,
        total_chunks=5,
        collection_size=5,
        is_healthy=True,
        last_update="2024-01-01T00:00:00Z",
    )

    # Mock Document
    mock_document = MagicMock(spec=Document)
    mock_document.id = 1
    mock_document.title = "Тестовый документ"
    mock_document.is_processed = True
    mock_document.owner_id = 1

    # Mock async service methods
    service.get_rag_status = AsyncMock(return_value=default_status)
    service.search_similar = AsyncMock(
        return_value=[
            ("Тестовый контекст", 0.9, {"document_title": "Тестовый документ"})
        ]
    )
    service.delete_document = AsyncMock(return_value=True)
    # Do not mock process_document to allow real method execution with mocked dependencies

    return service
