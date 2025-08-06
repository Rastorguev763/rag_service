import uuid
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config.config import settings
from app.rag.embedding import get_embedding_model
from app.utils.logger import logger


class QdrantVectorStore:
    """Класс для работы с векторной базой данных Qdrant"""

    def __init__(self, collection_name: Optional[str] = None):
        # Отключаем предупреждение о небезопасном соединении для локальной разработки
        import warnings

        warnings.filterwarnings(
            "ignore", message="Api key is used with an insecure connection"
        )

        self.client = AsyncQdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        self.collection_name = collection_name or settings.collection_name

    def get_user_collection_name(self, user_id: int) -> str:
        """Генерация имени коллекции для пользователя"""
        return f"user_{user_id}_collection"

    async def _ensure_collection_exists(self, collection_name: Optional[str] = None):
        """Проверка и создание коллекции если не существует"""
        try:
            target_collection = collection_name or self.collection_name
            collections = await self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if target_collection not in collection_names:
                logger.info(f"Создание коллекции: {target_collection}")
                # Получаем размерность модели эмбеддингов

                embedding_model = get_embedding_model()
                dimension = embedding_model.get_dimension()

                await self.client.create_collection(
                    collection_name=target_collection,
                    vectors_config=VectorParams(
                        size=dimension, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Коллекция {target_collection} создана")
            else:
                logger.info(f"Коллекция {target_collection} уже существует")

        except Exception as e:
            logger.error(f"Ошибка при создании коллекции: {e}")
            raise

    async def create_user_collection(self, user_id: int) -> str:
        """Создание персональной коллекции для пользователя"""
        try:
            collection_name = self.get_user_collection_name(user_id)
            await self._ensure_collection_exists(collection_name)
            logger.info(
                f"Создана персональная коллекция для пользователя {user_id}: {collection_name}"
            )
            return collection_name
        except Exception as e:
            logger.error(f"Ошибка при создании коллекции пользователя {user_id}: {e}")
            raise

    async def delete_user_collection(self, user_id: int) -> bool:
        """Удаление персональной коллекции пользователя"""
        try:
            collection_name = self.get_user_collection_name(user_id)
            await self.client.delete_collection(collection_name)
            logger.info(f"Удалена коллекция пользователя {user_id}: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Ошибка при удалении коллекции пользователя {user_id}: {e}")
            raise

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Добавление текстов в векторную базу"""
        try:
            target_collection = collection_name or self.collection_name
            await self._ensure_collection_exists(target_collection)

            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            if metadatas is None:
                metadatas = [{} for _ in texts]

            # Создаем эмбеддинги для текстов если не переданы
            if embeddings is None:
                from app.rag.embedding import get_embedding_model

                embedding_model = get_embedding_model()
                embeddings = await embedding_model.encode_async(texts)

            # Создаем точки для Qdrant
            points = []
            for i, (text, metadata, point_id, embedding) in enumerate(
                zip(texts, metadatas, ids, embeddings)
            ):
                # Убеждаемся, что embedding - это список float
                if hasattr(embedding, "tolist"):
                    vector = embedding.tolist()
                else:
                    vector = embedding

                point = PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": text, "metadata": metadata},
                )
                points.append(point)

            # Добавляем точки в коллекцию
            await self.client.upsert(collection_name=target_collection, points=points)

            logger.info(
                f"Добавлено {len(points)} точек в коллекцию {target_collection}"
            )
            return ids

        except Exception as e:
            logger.error(f"Ошибка при добавлении текстов: {e}")
            raise

    async def add_user_message(
        self,
        user_id: int,
        message_text: str,
        metadata: Optional[Dict[str, Any]] = None,
        message_id: Optional[str] = None,
    ) -> str:
        """Добавление сообщения пользователя в его персональную коллекцию"""
        try:
            collection_name = self.get_user_collection_name(user_id)

            # Добавляем информацию о пользователе в метаданные
            if metadata is None:
                metadata = {}
            metadata.update(
                {
                    "user_id": user_id,
                    "message_type": "user_message",
                    "timestamp": datetime.now(UTC).isoformat(),
                }
            )

            # Генерируем ID если не передан
            if message_id is None:
                message_id = str(uuid.uuid4())

            # Добавляем сообщение в коллекцию
            await self.add_texts(
                texts=[message_text],
                metadatas=[metadata],
                ids=[message_id],
                collection_name=collection_name,
            )

            logger.info(f"Добавлено сообщение пользователя {user_id} в коллекцию")
            return message_id

        except Exception as e:
            logger.error(f"Ошибка при добавлении сообщения пользователя: {e}")
            raise

    async def similarity_search(
        self,
        query: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Поиск похожих документов"""
        try:
            # Проверяем, что k >= 1
            k = max(1, k)

            target_collection = collection_name or self.collection_name
            # Убеждаемся, что коллекция существует
            await self._ensure_collection_exists(target_collection)

            # Создаем фильтр если указан
            search_filter = None
            if filter_metadata:
                conditions = []
                for key, value in filter_metadata.items():
                    conditions.append(
                        FieldCondition(
                            key=f"metadata.{key}", match=MatchValue(value=value)
                        )
                    )
                search_filter = Filter(must=conditions)

            # Выполняем поиск
            search_result = await self.client.search(
                collection_name=target_collection,
                query_vector=query,
                limit=k,
                query_filter=search_filter,
                with_payload=True,
            )

            # Формируем результат
            results = []
            for result in search_result:
                text = result.payload.get("text", "")
                metadata = result.payload.get("metadata", {})
                score = result.score
                results.append((text, score, metadata))

            logger.debug(
                f"Найдено {len(results)} похожих документов в коллекции {target_collection}"
            )
            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise

    async def search_user_messages(
        self,
        user_id: int,
        query: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Поиск в персональной коллекции пользователя"""
        try:
            # Проверяем, что k >= 1
            k = max(1, k)

            collection_name = self.get_user_collection_name(user_id)

            # Добавляем фильтр по пользователю
            if filter_metadata is None:
                filter_metadata = {}
            filter_metadata["user_id"] = user_id

            return await self.similarity_search(
                query=query,
                k=k,
                filter_metadata=filter_metadata,
                collection_name=collection_name,
            )
        except Exception as e:
            logger.error(f"Ошибка при поиске сообщений пользователя {user_id}: {e}")
            raise

    async def delete_texts(
        self, ids: List[str], collection_name: Optional[str] = None
    ) -> bool:
        """Удаление текстов по ID"""
        try:
            target_collection = collection_name or self.collection_name
            await self.client.delete(
                collection_name=target_collection, points_selector=ids
            )
            logger.info(f"Удалено {len(ids)} точек из коллекции {target_collection}")
            return True

        except Exception as e:
            logger.error(f"Ошибка при удалении текстов: {e}")
            raise

    async def get_collection_info(
        self, collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Получение информации о коллекции"""
        try:
            target_collection = collection_name or self.collection_name
            info = await self.client.get_collection(target_collection)
            return {
                "name": target_collection,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}")
            raise

    async def get_user_collection_info(self, user_id: int) -> Dict[str, Any]:
        """Получение информации о персональной коллекции пользователя"""
        try:
            collection_name = self.get_user_collection_name(user_id)
            return await self.get_collection_info(collection_name)
        except Exception as e:
            logger.error(
                f"Ошибка при получении информации о коллекции пользователя {user_id}: {e}"
            )
            raise

    async def clear_collection(self, collection_name: Optional[str] = None) -> bool:
        """Очистка коллекции"""
        try:
            target_collection = collection_name or self.collection_name
            await self.client.delete_collection(target_collection)
            await self._ensure_collection_exists(target_collection)
            logger.info(f"Коллекция {target_collection} очищена")
            return True
        except Exception as e:
            logger.error(f"Ошибка при очистке коллекции: {e}")
            raise

    async def list_user_collections(self) -> List[str]:
        """Получение списка всех пользовательских коллекций"""
        try:
            collections = await self.client.get_collections()
            user_collections = []
            for collection in collections.collections:
                if collection.name.startswith("user_") and collection.name.endswith(
                    "_collection"
                ):
                    user_collections.append(collection.name)
            return user_collections
        except Exception as e:
            logger.error(f"Ошибка при получении списка пользовательских коллекций: {e}")
            raise


def get_vector_store(collection_name: Optional[str] = None) -> QdrantVectorStore:
    return QdrantVectorStore(collection_name)


def get_user_vector_store(user_id: int) -> QdrantVectorStore:
    """Получение векторного хранилища для конкретного пользователя"""
    collection_name = f"user_{user_id}_collection"
    return QdrantVectorStore(collection_name)
