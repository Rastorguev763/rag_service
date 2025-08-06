from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.models import Document, DocumentChunk, User
from app.rag.embedding import get_embedding_model
from app.rag.text_splitter import get_text_splitter
from app.rag.vectorstore import get_vector_store
from app.schemas.schemas import DocumentCreate, RAGStatus
from app.utils.logger import logger


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
                    "content_type": "document_chunk",
                    "created_at": datetime.now(UTC).isoformat(),
                }
                metadatas.append(metadata)

            # Добавляем в векторную базу (в общую коллекцию документов)
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

    async def add_user_message(
        self, user_id: int, message_text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Добавление сообщения пользователя в его персональную коллекцию"""
        try:
            # Добавляем сообщение в персональную коллекцию пользователя
            message_id = await self.vector_store.add_user_message(
                user_id=user_id, message_text=message_text, metadata=metadata
            )

            logger.info(
                f"Добавлено сообщение пользователя {user_id} в персональную коллекцию"
            )
            return message_id

        except Exception as e:
            logger.error(f"Ошибка при добавлении сообщения пользователя: {e}")
            raise

    async def search_similar(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[int] = None,
        search_user_messages: bool = False,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Поиск похожих документов"""
        try:
            # Создаем эмбеддинг для запроса
            query_embedding = await self.embedding_model.encode_single_async(query)

            if search_user_messages and user_id:
                # Поиск в персональной коллекции пользователя
                results = await self.vector_store.search_user_messages(
                    user_id=user_id, query=query_embedding, k=k
                )
            else:
                # Фильтр по пользователю если указан
                filter_metadata = None
                if user_id:
                    filter_metadata = {"user_id": user_id}

                # Ищем похожие документы в общей коллекции
                results = await self.vector_store.similarity_search(
                    query=query_embedding, k=k, filter_metadata=filter_metadata
                )

            logger.debug(f"Найдено {len(results)} похожих документов")
            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise

    async def search_hybrid(
        self,
        query: str,
        k: int = 5,
        user_id: Optional[int] = None,
        user_messages_weight: float = 0.3,
        documents_weight: float = 0.7,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Гибридный поиск по документам и сообщениям пользователя"""
        try:
            if not user_id:
                # Если пользователь не указан, ищем только в документах
                return await self.search_similar(
                    query, k, user_id, search_user_messages=False
                )

            # Создаем эмбеддинг для запроса
            query_embedding = await self.embedding_model.encode_single_async(query)

            # Поиск в документах пользователя
            document_k = max(1, int(k * documents_weight))
            document_results = await self.vector_store.similarity_search(
                query=query_embedding,
                k=document_k,
                filter_metadata={"user_id": user_id, "content_type": "document_chunk"},
            )

            # Поиск в сообщениях пользователя
            message_k = max(1, int(k * user_messages_weight))
            message_results = await self.vector_store.search_user_messages(
                user_id=user_id, query=query_embedding, k=message_k
            )

            # Объединяем и сортируем результаты
            all_results = []

            # Добавляем результаты документов с весовым коэффициентом
            for text, score, metadata in document_results:
                all_results.append((text, score * documents_weight, metadata))

            # Добавляем результаты сообщений с весовым коэффициентом
            for text, score, metadata in message_results:
                all_results.append((text, score * user_messages_weight, metadata))

            # Сортируем по релевантности и берем топ k
            all_results.sort(key=lambda x: x[1], reverse=True)
            final_results = all_results[:k]

            logger.debug(f"Гибридный поиск: найдено {len(final_results)} результатов")
            return final_results

        except Exception as e:
            logger.error(f"Ошибка при гибридном поиске: {e}")
            raise

    async def get_user_collection_info(self, user_id: int) -> Dict[str, Any]:
        """Получение информации о персональной коллекции пользователя"""
        try:
            collection_info = await self.vector_store.get_user_collection_info(user_id)
            return collection_info
        except Exception as e:
            logger.error(
                f"Ошибка при получении информации о коллекции пользователя: {e}"
            )
            raise

    async def get_rag_status(
        self, db: AsyncSession, user_id: Optional[int] = None
    ) -> RAGStatus:
        """Получение статуса RAG системы"""
        try:
            # Статистика из PostgreSQL
            if user_id:
                result = await db.execute(
                    select(Document).filter(Document.owner_id == user_id)
                )
                total_documents = len(result.scalars().all())

                result = await db.execute(
                    select(Document).filter(
                        Document.owner_id == user_id, Document.is_processed.is_(True)
                    )
                )
                processed_documents = len(result.scalars().all())

                result = await db.execute(
                    select(DocumentChunk)
                    .join(Document)
                    .filter(Document.owner_id == user_id)
                )
                total_chunks = len(result.scalars().all())
            else:
                result = await db.execute(select(Document))
                total_documents = len(result.scalars().all())

                result = await db.execute(
                    select(Document).filter(Document.is_processed.is_(True))
                )
                processed_documents = len(result.scalars().all())

                result = await db.execute(select(DocumentChunk))
                total_chunks = len(result.scalars().all())

            # Статистика из Qdrant
            if user_id:
                collection_info = await self.vector_store.get_user_collection_info(
                    user_id
                )
                collection_size = collection_info.get("points_count", 0)
            else:
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

    async def clear_user_data(self, user_id: int, db: AsyncSession) -> bool:
        """Очистка всех данных пользователя"""
        try:
            # Удаляем все документы пользователя
            result = await db.execute(
                select(Document).filter(Document.owner_id == user_id)
            )
            user_documents = result.scalars().all()

            for document in user_documents:
                await self.delete_document(
                    db, document.id, User(id=user_id, username="temp")
                )

            # Удаляем персональную коллекцию пользователя
            await self.vector_store.delete_user_collection(user_id)

            logger.info(f"Очищены все данные пользователя {user_id}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при очистке данных пользователя: {e}")
            raise


def get_rag_service() -> RAGService:
    return RAGService()
