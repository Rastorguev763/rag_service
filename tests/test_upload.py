"""
Тесты для endpoints загрузки документов
"""

import pytest
from fastapi import status
from unittest.mock import patch
import io


@pytest.mark.asyncio
class TestUploadEndpoints:
    """Тесты для endpoints загрузки документов"""

    async def test_upload_text_document_success(self, client, auth_headers, mock_rag_service):
        """Тест успешной загрузки текстового документа"""
        document_data = {
            "title": "Тестовый документ",
            "content": "Это тестовый документ для проверки RAG системы.",
            "chunk_size": 1000,
            "chunk_overlap": 200,
        }

        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.post(
                "/upload/text", json=document_data, headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["title"] == document_data["title"]
            assert data["status"] == "processed"
            assert "document_id" in data

    async def test_upload_text_document_unauthorized(self, client, mock_rag_service):
        """Тест загрузки текстового документа без аутентификации"""
        document_data = {
            "title": "Тестовый документ",
            "content": "Это тестовый документ для проверки RAG системы.",
        }

        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.post("/upload/text", json=document_data)

            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_upload_text_document_invalid_data(self, client, auth_headers):
        """Тест загрузки текстового документа с некорректными данными"""
        document_data = {
            "title": "",  # Пустой заголовок
            "content": "",  # Пустое содержимое
        }

        response = client.post("/upload/text", json=document_data, headers=auth_headers)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_upload_text_document_service_error(
        self, client, auth_headers, mock_rag_service
    ):
        """Тест загрузки текстового документа при ошибке сервиса"""
        document_data = {
            "title": "Тестовый документ",
            "content": "Это тестовый документ для проверки RAG системы.",
        }

        # Симулируем ошибку в RAG сервисе
        mock_rag_service.process_document.side_effect = Exception("Processing failed")

        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.post(
                "/upload/text", json=document_data, headers=auth_headers
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    async def test_upload_file_success(self, client, auth_headers, mock_rag_service):
        """Тест успешной загрузки файла"""
        file_content = "Это тестовый файл для проверки RAG системы."
        file_data = {
            "file": ("test.txt", io.BytesIO(file_content.encode()), "text/plain"),
            "title": "Тестовый файл",
            "chunk_size": "1000",
            "chunk_overlap": "200",
        }

        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.post(
                "/upload/file", files=file_data, headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["title"] == "Тестовый файл"
            assert data["status"] == "processed"

    async def test_upload_file_unsupported_type(self, client, auth_headers):
        """Тест загрузки файла неподдерживаемого типа"""
        file_content = b"test content"
        file_data = {
            "file": ("test.exe", io.BytesIO(file_content), "application/octet-stream"),
            "title": "Тестовый файл",
        }

        response = client.post("/upload/file", files=file_data, headers=auth_headers)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not supported" in response.json()["detail"]

    async def test_upload_file_pdf_not_implemented(self, client, auth_headers):
        """Тест загрузки PDF файла (пока не реализовано)"""
        file_content = b"%PDF-1.4 test pdf content"
        file_data = {
            "file": ("test.pdf", io.BytesIO(file_content), "application/pdf"),
            "title": "Тестовый PDF",
        }

        response = client.post("/upload/file", files=file_data, headers=auth_headers)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "not implemented" in response.json()["detail"]

    async def test_upload_file_unauthorized(self, client, mock_rag_service):
        """Тест загрузки файла без аутентификации"""
        file_content = "Это тестовый файл."
        file_data = {
            "file": ("test.txt", io.BytesIO(file_content.encode()), "text/plain"),
            "title": "Тестовый файл",
        }

        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.post("/upload/file", files=file_data)

            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_upload_file_service_error(self, client, auth_headers, mock_rag_service):
        """Тест загрузки файла при ошибке сервиса"""
        file_content = "Это тестовый файл."
        file_data = {
            "file": ("test.txt", io.BytesIO(file_content.encode()), "text/plain"),
            "title": "Тестовый файл",
        }

        # Симулируем ошибку в RAG сервисе
        mock_rag_service.process_document.side_effect = Exception(
            "File processing failed"
        )

        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.post(
                "/upload/file", files=file_data, headers=auth_headers
            )

            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    async def test_get_documents_success(self, client, auth_headers, test_document):
        """Тест получения списка документов пользователя"""
        response = client.get("/upload/documents", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
        assert data[0]["title"] == test_document.title

    async def test_get_documents_unauthorized(self, client):
        """Тест получения документов без аутентификации"""
        response = client.get("/upload/documents")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_document_success(self, client, auth_headers, test_document):
        """Тест получения конкретного документа"""
        response = client.get(
            f"/upload/documents/{test_document.id}", headers=auth_headers
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["id"] == test_document.id
        assert data["title"] == test_document.title

    async def test_get_document_not_found(self, client, auth_headers):
        """Тест получения несуществующего документа"""
        response = client.get("/upload/documents/99999", headers=auth_headers)

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_get_document_unauthorized(self, client, test_document):
        """Тест получения документа без аутентификации"""
        response = client.get(f"/upload/documents/{test_document.id}")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_document_wrong_owner(self, client, auth_headers_user2, test_document):
        """Тест получения документа другого пользователя"""
        response = client.get(
            f"/upload/documents/{test_document.id}", headers=auth_headers_user2
        )

        assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_document_success(
        self, client, auth_headers, test_document, mock_rag_service
    ):
        """Тест успешного удаления документа"""
        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.delete(
                f"/upload/documents/{test_document.id}", headers=auth_headers
            )

            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["success"] is True

    async def test_delete_document_not_found(self, client, auth_headers, mock_rag_service):
        """Тест удаления несуществующего документа"""
        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.delete("/upload/documents/99999", headers=auth_headers)

            assert response.status_code == status.HTTP_404_NOT_FOUND

    async def test_delete_document_unauthorized(self, client, test_document):
        """Тест удаления документа без аутентификации"""
        response = client.delete(f"/upload/documents/{test_document.id}")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_delete_document_wrong_owner(
        self, client, auth_headers_user2, test_document, mock_rag_service
    ):
        """Тест удаления документа другого пользователя"""
        with patch(
            "app.api.endpoints.upload.get_rag_service", return_value=mock_rag_service
        ):
            response = client.delete(
                f"/upload/documents/{test_document.id}", headers=auth_headers_user2
            )

            assert response.status_code == status.HTTP_404_NOT_FOUND
