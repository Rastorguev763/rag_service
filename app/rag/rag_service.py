from datetime import datetime, UTC
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.utils.logger import logger
from app.models.models import Document, DocumentChunk, User
from app.rag.embedding import get_embedding_model
from app.rag.text_splitter import get_text_splitter
from app.rag.vectorstore import get_vector_store
from app.schemas.schemas import DocumentCreate, RAGStatus


class RAGService:
    """Основной сервис RAG"""

    def __init__(self):
        self.embedding_model = get_embedding_model()
        self.vector_store = get_vector_store()
        self.text_splitter = get_text_splitter()

    async def process_document(
        self, db: AsyncSession, document_data: DocumentCreate, user: User
    ) -> Document:
        """Обработка документа и добавление в RAG"""
        try:
            logger.info(f"Начало обработки документа: {document_data.title}")

            # Создаем документ в БД
            document = Document(
                title=document_data.title,
                content=document_data.content,
                file_path=document_data.file_path,
                file_type=document_data.file_type,
                chunk_size=document_data.chunk_size,
                chunk_overlap=document_data.chunk_overlap,
                owner_id=user.id,
                is_processed=False,
            )
            db.add(document)
            await db.commit()
            await db.refresh(document)

            # Разбиваем текст на чанки
            chunks = self.text_splitter.split_text(document_data.content)
            logger.info(f"Создано {len(chunks)} чанков")

            # Создаем эмбеддинги для чанков
            embeddings = await self.embedding_model.encode_async(chunks)

            # Подготавливаем метаданные для каждого чанка
            metadatas = []
            for i, chunk in enumerate(chunks):
                metadata = {
                    "document_id": document.id,
                    "document_title": document.title,
                    "chunk_index": i,
                    "user_id": user.id,
                    "created_at": datetime.now(UTC).isoformat(),
                }
                metadatas.append(metadata)

            # Добавляем в векторную базу
            chunk_ids = await self.vector_store.add_texts(
                texts=chunks, metadatas=metadatas, embeddings=embeddings
            )

            # Сохраняем чанки в БД
            for i, (chunk, embedding_id) in enumerate(zip(chunks, chunk_ids)):
                db_chunk = DocumentChunk(
                    content=chunk,
                    chunk_index=i,
                    embedding_id=embedding_id,
                    document_id=document.id,
                )
                db.add(db_chunk)

            # Обновляем статус документа
            document.is_processed = True
            await db.commit()

            logger.info(f"Документ {document.id} успешно обработан")
            return document

        except Exception as e:
            logger.error(f"Ошибка при обработке документа: {e}")
            await db.rollback()
            raise

    async def search_similar(
        self, query: str, k: int = 5, user_id: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Поиск похожих документов"""
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = await self.embedding_model.encode_single_async(query)

            # Фильтр по пользователю если указан
            filter_metadata = None
            if user_id:
                filter_metadata = {"user_id": user_id}

            # Ищем похожие документы
            results = await self.vector_store.similarity_search(
                query=query_embedding, k=k, filter_metadata=filter_metadata
            )

            logger.debug(f"Найдено {len(results)} похожих документов")
            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise

    async def get_rag_status(self, db: AsyncSession) -> RAGStatus:
        """Получение статуса RAG системы"""
        try:
            # Статистика из PostgreSQL
            result = await db.execute(select(Document))
            total_documents = len(result.scalars().all())

            result = await db.execute(
                select(Document).filter(Document.is_processed.is_(True))
            )
            processed_documents = len(result.scalars().all())

            result = await db.execute(select(DocumentChunk))
            total_chunks = len(result.scalars().all())

            # Статистика из Qdrant
            collection_info = await self.vector_store.get_collection_info()
            collection_size = collection_info.get("points_count", 0)

            # Проверка здоровья системы
            is_healthy = (
                total_documents > 0
                and processed_documents > 0
                and total_chunks > 0
                and collection_size > 0
            )

            return RAGStatus(
                total_documents=total_documents,
                processed_documents=processed_documents,
                total_chunks=total_chunks,
                collection_size=collection_size,
                is_healthy=is_healthy,
                last_update=datetime.now(UTC).isoformat(),
            )

        except Exception as e:
            logger.error(f"Ошибка при получении статуса RAG: {e}")
            raise

    async def delete_document(
        self, db: AsyncSession, document_id: int, user: User
    ) -> bool:
        """Удаление документа из RAG"""
        try:
            # Получаем документ
            result = await db.execute(
                select(Document).filter(
                    Document.id == document_id, Document.owner_id == user.id
                )
            )
            document = result.scalar_one_or_none()

            if not document:
                logger.warning(f"Документ {document_id} не найден")
                return False

            # Получаем ID эмбеддингов для удаления
            chunk_ids = [
                chunk.embedding_id for chunk in document.chunks if chunk.embedding_id
            ]

            # Удаляем из векторной базы
            if chunk_ids:
                await self.vector_store.delete_texts(chunk_ids)

            # Удаляем из БД
            await db.delete(document)
            await db.commit()

            logger.info(f"Документ {document_id} удален")
            return True

        except Exception as e:
            logger.error(f"Ошибка при удалении документа: {e}")
            await db.rollback()
            raise


def get_rag_service() -> RAGService:
    return RAGService()
