"""
Тесты для схем валидации
"""

import pytest
from pydantic import ValidationError

from app.schemas.schemas import ChatRequest, ChatResponse, UserCreate, DocumentCreate


@pytest.mark.asyncio
class TestChatRequest:
    """Тесты для схемы ChatRequest"""

    async def test_valid_chat_request(self):
        """Тест валидного запроса чата"""
        data = {
            "message": "Тестовое сообщение",
            "use_rag": True,
            "k_points": 5,
            "max_tokens": 1000,
        }

        request = ChatRequest(**data)

        assert request.message == data["message"]
        assert request.use_rag == data["use_rag"]
        assert request.k_points == data["k_points"]
        assert request.max_tokens == data["max_tokens"]

    async def test_chat_request_defaults(self):
        """Тест значений по умолчанию"""
        data = {"message": "Тестовое сообщение"}

        request = ChatRequest(**data)

        assert request.use_rag is True
        assert request.k_points == 3  # Значение по умолчанию из settings
        assert request.max_tokens == 1000

    async def test_chat_request_k_points_validation_min(self):
        """Тест валидации k_points (минимум)"""
        data = {"message": "Тестовое сообщение", "k_points": 0}

        with pytest.raises(ValidationError):
            ChatRequest(**data)

    async def test_chat_request_k_points_validation_max(self):
        """Тест валидации k_points (максимум)"""
        data = {"message": "Тестовое сообщение", "k_points": 25}

        with pytest.raises(ValidationError):
            ChatRequest(**data)

    async def test_chat_request_max_tokens_validation_min(self):
        """Тест валидации max_tokens (минимум)"""
        data = {"message": "Тестовое сообщение", "max_tokens": 0}

        with pytest.raises(ValidationError):
            ChatRequest(**data)

    async def test_chat_request_max_tokens_validation_max(self):
        """Тест валидации max_tokens (максимум)"""
        data = {"message": "Тестовое сообщение", "max_tokens": 5000}

        with pytest.raises(ValidationError):
            ChatRequest(**data)

    async def test_chat_request_empty_message(self):
        """Тест с пустым сообщением"""
        data = {"message": ""}

        with pytest.raises(ValidationError):
            ChatRequest(**data)

    async def test_chat_request_missing_message(self):
        """Тест без сообщения"""
        data = {}

        with pytest.raises(ValidationError):
            ChatRequest(**data)


@pytest.mark.asyncio
class TestChatResponse:
    """Тесты для схемы ChatResponse"""

    async def test_valid_chat_response(self):
        """Тест валидного ответа чата"""
        data = {
            "message": "Ответ ассистента",
            "session_id": 1,
            "message_id": 1,
            "sources": ["Документ 1", "Документ 2"],
            "k_points_used": 2,
        }

        response = ChatResponse(**data)

        assert response.message == data["message"]
        assert response.session_id == data["session_id"]
        assert response.message_id == data["message_id"]
        assert response.sources == data["sources"]
        assert response.k_points_used == data["k_points_used"]

    async def test_chat_response_without_optional_fields(self):
        """Тест ответа чата без опциональных полей"""
        data = {"message": "Ответ ассистента", "session_id": 1, "message_id": 1}

        response = ChatResponse(**data)

        assert response.sources is None
        assert response.k_points_used is None


@pytest.mark.asyncio
class TestUserCreate:
    """Тесты для схемы UserCreate"""

    async def test_valid_user_create(self):
        """Тест валидного создания пользователя"""
        data = {
            "username": "testuser",
            "email": "test@example.com",
            "password": "password123",
        }

        user = UserCreate(**data)

        assert user.username == data["username"]
        assert user.email == data["email"]
        assert user.password == data["password"]

    async def test_user_create_invalid_email(self):
        """Тест создания пользователя с некорректным email"""
        data = {
            "username": "testuser",
            "email": "invalid-email",
            "password": "password123",
        }

        with pytest.raises(ValidationError):
            UserCreate(**data)

    async def test_user_create_short_password(self):
        """Тест создания пользователя с коротким паролем"""
        data = {"username": "testuser", "email": "test@example.com", "password": "123"}

        # В схеме UserCreate нет валидации длины пароля, поэтому тест должен проходить
        user = UserCreate(**data)
        assert user.password == "123"


@pytest.mark.asyncio
class TestDocumentCreate:
    """Тесты для схемы DocumentCreate"""

    async def test_valid_document_create(self):
        """Тест валидного создания документа"""
        data = {
            "title": "Тестовый документ",
            "content": "Содержимое документа",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        }

        document = DocumentCreate(**data)

        assert document.title == data["title"]
        assert document.content == data["content"]
        assert document.chunk_size == data["chunk_size"]
        assert document.chunk_overlap == data["chunk_overlap"]

    async def test_document_create_defaults(self):
        """Тест значений по умолчанию"""
        data = {"title": "Тестовый документ", "content": "Содержимое документа"}

        document = DocumentCreate(**data)

        assert document.chunk_size == 1000
        assert document.chunk_overlap == 200

    async def test_document_create_empty_title(self):
        """Тест создания документа с пустым заголовком"""
        data = {"title": "", "content": "Содержимое документа"}

        # В схеме DocumentCreate нет валидации на пустые строки, поэтому тест должен проходить
        document = DocumentCreate(**data)
        assert document.title == ""

    async def test_document_create_empty_content(self):
        """Тест создания документа с пустым содержимым"""
        data = {"title": "Тестовый документ", "content": ""}

        # В схеме DocumentCreate нет валидации на пустые строки, поэтому тест должен проходить
        document = DocumentCreate(**data)
        assert document.content == ""
