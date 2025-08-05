import pytest
from fastapi import status
from unittest.mock import patch

from app.main import app  # Импорт на уровне модуля
from app.rag.llm import get_llm  # Импорт на уровне модуля


@pytest.mark.asyncio
@pytest.mark.chat
class TestChatEndpoints:
    """Тесты для endpoints чата"""

    async def test_chat_with_rag_success(
        self, client, auth_headers, mock_rag_service, mock_llm
    ):
        """Тест успешного чата с RAG"""
        chat_data = {
            "message": "Расскажи о RAG системе",
            "use_rag": True,
            "k_points": 3,
            "max_tokens": 1000,
        }

        # Переопределяем зависимость LLM
        app.dependency_overrides[get_llm] = lambda: mock_llm

        try:
            with patch(
                "app.api.endpoints.chat.get_rag_service", return_value=mock_rag_service
            ):
                response = client.post(
                    "/chat", json=chat_data, headers=auth_headers
                )

                assert response.status_code == status.HTTP_200_OK
                data = response.json()
                assert "message" in data
                assert "session_id" in data
                assert "message_id" in data
                assert data["message"] == "Это тестовый ответ от LLM"
        finally:
            # Очищаем переопределение
            app.dependency_overrides.pop(get_llm, None)

    async def test_chat_without_rag_success(self, client, auth_headers, mock_llm):
        """Тест успешного чата без RAG"""
        chat_data = {
            "message": "Привет, как дела?",
            "use_rag": False,
            "max_tokens": 500,
        }

        # Переопределяем зависимость LLM
        app.dependency_overrides[get_llm] = lambda: mock_llm

        try:
            response = client.post("/chat", json=chat_data, headers=auth_headers)

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert "message" in data
            assert data["sources"] is None
            assert data["k_points_used"] is None
        finally:
            # Очищаем переопределение
            app.dependency_overrides.pop(get_llm, None)

    async def test_chat_unauthorized(self, client, mock_rag_service, mock_llm):
        """Тест чата без аутентификации"""
        chat_data = {"message": "Тестовое сообщение", "use_rag": True}

        # Переопределяем зависимость LLM
        app.dependency_overrides[get_llm] = lambda: mock_llm

        try:
            with patch(
                "app.api.endpoints.chat.get_rag_service", return_value=mock_rag_service
            ):
                response = client.post("/chat", json=chat_data)

                assert response.status_code == status.HTTP_401_UNAUTHORIZED
        finally:
            # Очищаем переопределение
            app.dependency_overrides.pop(get_llm, None)

    async def test_chat_invalid_data(self, client, auth_headers):
        """Тест чата с некорректными данными"""
        chat_data = {"message": "", "use_rag": True}  # Пустое сообщение

        response = client.post("/chat", json=chat_data, headers=auth_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_chat_k_points_validation(
        self, client, auth_headers, mock_rag_service, mock_llm
    ):
        """Тест валидации k_points"""
        chat_data = {"message": "Тестовое сообщение", "use_rag": True, "k_points": 1}

        # Переопределяем зависимость LLM
        app.dependency_overrides[get_llm] = lambda: mock_llm

        try:
            with patch(
                "app.api.endpoints.chat.get_rag_service", return_value=mock_rag_service
            ):
                response = client.post(
                    "/chat", json=chat_data, headers=auth_headers
                )

                assert response.status_code == status.HTTP_200_OK
        finally:
            # Очищаем переопределение
            app.dependency_overrides.pop(get_llm, None)

    async def test_get_chat_sessions_success(
        self, client, auth_headers, test_chat_session
    ):
        """Тест получения списка сессий чата"""
        response = client.get("/chat/sessions", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    async def test_get_chat_sessions_unauthorized(self, client):
        """Тест получения сессий чата без аутентификации"""
        response = client.get("/chat/sessions")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_chat_messages_success(
        self, client, auth_headers, test_chat_session, test_chat_message
    ):
        """Тест получения сообщений сессии"""
        response = client.get(
            f"/chat/sessions/{test_chat_session.id}/messages", headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["content"] == test_chat_message.content

    async def test_delete_chat_session_success(
        self, client, auth_headers, test_chat_session
    ):
        """Тест успешного удаления сессии чата"""
        response = client.delete(
            f"/chat/sessions/{test_chat_session.id}", headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "Chat session deleted"
