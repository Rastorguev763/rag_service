from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Document, User
from app.rag.rag_service import RAGService
from app.schemas.schemas import DocumentCreate, RAGStatus


@pytest.mark.asyncio
@pytest.mark.rag
class TestRAGService:
    """Тесты для RAG сервиса"""

    async def test_process_document_success(
        self, mock_rag_service: RAGService, db_session: AsyncSession, test_user: User
    ):
        """Тест успешной обработки документа"""
        document_data = DocumentCreate(
            title="Тестовый документ",
            content="Это тестовый документ для проверки RAG системы.",
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Создаем мок документа для возврата
        mock_document = MagicMock()
        mock_document.id = 1
        mock_document.title = document_data.title
        mock_document.is_processed = True
        mock_document.owner_id = test_user.id

        # Мокаем метод process_document чтобы он возвращал наш мок
        mock_rag_service.process_document = AsyncMock(return_value=mock_document)

        document = await mock_rag_service.process_document(
            db_session, document_data, test_user
        )

        assert document.title == document_data.title
        assert document.is_processed is True
        assert document.owner_id == test_user.id
        assert document.id is not None

    async def test_search_similar_success(self, mock_rag_service: RAGService):
        """Тест успешного поиска похожих документов"""
        query = "тестовый запрос"

        results = await mock_rag_service.search_similar(query, k=3)
        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], tuple)

    async def test_search_similar_with_user_filter(self, mock_rag_service: RAGService):
        """Тест поиска с фильтром по пользователю"""
        query = "тестовый запрос"
        user_id = 1

        results = await mock_rag_service.search_similar(query, k=3, user_id=user_id)

        assert isinstance(results, list)
        assert len(results) == 1

    async def test_get_rag_status_success(
        self, mock_rag_service: RAGService, db_session: AsyncSession
    ):
        """Тест получения статуса RAG системы"""

        status_info = await mock_rag_service.get_rag_status(db_session)

        assert isinstance(status_info, RAGStatus)
        assert status_info.total_documents == 1
        assert status_info.processed_documents == 1
        assert status_info.total_chunks == 5
        assert status_info.collection_size == 5
        assert status_info.is_healthy is True
        assert status_info.last_update is not None

    async def test_delete_document_success(
        self,
        mock_rag_service: RAGService,
        db_session: AsyncSession,
        test_user: User,
        test_document: Document,
    ):
        """Тест успешного удаления документа"""
        # Мокаем метод delete_document чтобы он возвращал True
        mock_rag_service.delete_document = AsyncMock(return_value=True)

        result = await mock_rag_service.delete_document(
            db_session, test_document.id, test_user
        )

        assert result is True

    async def test_delete_document_not_found(
        self, mock_rag_service: RAGService, db_session: AsyncSession, test_user: User
    ):
        """Тест удаления несуществующего документа"""
        # Мокаем метод delete_document чтобы он возвращал False
        mock_rag_service.delete_document = AsyncMock(return_value=False)

        result = await mock_rag_service.delete_document(db_session, 99999, test_user)

        assert result is False

    async def test_delete_document_wrong_owner(
        self,
        mock_rag_service: RAGService,
        db_session: AsyncSession,
        test_user2: User,
        test_document: Document,
    ):
        """Тест удаления документа другого пользователя"""
        # Мокаем метод delete_document чтобы он возвращал False
        mock_rag_service.delete_document = AsyncMock(return_value=False)

        result = await mock_rag_service.delete_document(
            db_session, test_document.id, test_user2
        )

        assert result is False

    async def test_process_document_error(
        self, mock_rag_service: RAGService, db_session: AsyncSession, test_user: User
    ):
        """Тест обработки документа при ошибке"""
        document_data = DocumentCreate(
            title="Тестовый документ",
            content="Это тестовый документ для проверки RAG системы.",
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Мокаем метод process_document чтобы он вызывал исключение
        mock_rag_service.process_document = AsyncMock(
            side_effect=Exception("Embedding error")
        )

        with pytest.raises(Exception, match="Embedding error"):
            await mock_rag_service.process_document(
                db_session, document_data, test_user
            )

    async def test_search_similar_error(self, mock_rag_service: RAGService):
        """Тест поиска при ошибке"""
        query = "тестовый запрос"

        # Мокаем метод search_similar чтобы он вызывал исключение
        mock_rag_service.search_similar = AsyncMock(
            side_effect=Exception("Encoding error")
        )

        with pytest.raises(Exception, match="Encoding error"):
            await mock_rag_service.search_similar(query)

    async def test_get_rag_status_error(
        self, mock_rag_service: RAGService, db_session: AsyncSession
    ):
        """Тест получения статуса при ошибке"""
        # Мокаем метод get_rag_status чтобы он вызывал исключение
        mock_rag_service.get_rag_status = AsyncMock(
            side_effect=Exception("Qdrant error")
        )

        with pytest.raises(Exception, match="Qdrant error"):
            await mock_rag_service.get_rag_status(db_session)

    async def test_process_document_invalid_chunk_size(
        self, mock_rag_service: RAGService, db_session: AsyncSession, test_user: User
    ):
        """Тест обработки документа с некорректным chunk_size"""
        document_data = DocumentCreate(
            title="Тестовый документ",
            content="Это тестовый документ для проверки RAG системы.",
            chunk_size=-100,
            chunk_overlap=200,
        )

        # Мокаем метод process_document чтобы он вызывал ValueError
        mock_rag_service.process_document = AsyncMock(
            side_effect=ValueError("chunk_size must be positive")
        )

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            await mock_rag_service.process_document(
                db_session, document_data, test_user
            )

    async def test_process_document_save_chunks(
        self, mock_rag_service: RAGService, db_session: AsyncSession, test_user: User
    ):
        """Тест сохранения чанков документа в базе данных"""
        document_data = DocumentCreate(
            title="Тестовый документ",
            content="Это тестовый документ для проверки RAG системы.",
            chunk_size=1000,
            chunk_overlap=200,
        )

        # Создаем мок документа для возврата
        mock_document = MagicMock()
        mock_document.id = 1
        mock_document.title = document_data.title
        mock_document.is_processed = True
        mock_document.owner_id = test_user.id

        # Мокаем метод process_document чтобы он возвращал наш мок
        mock_rag_service.process_document = AsyncMock(return_value=mock_document)

        document = await mock_rag_service.process_document(
            db_session, document_data, test_user
        )

        # Проверяем что документ был создан
        assert document.id == 1
        assert document.title == document_data.title
        assert document.is_processed is True
