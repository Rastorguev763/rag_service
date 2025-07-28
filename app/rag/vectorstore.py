import uuid
from typing import Any, Dict, List, Optional, Tuple

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.config.config import settings
from app.utils.logger import logger


class QdrantVectorStore:
    """Класс для работы с векторной базой данных Qdrant"""

    def __init__(self):
        # Отключаем предупреждение о небезопасном соединении для локальной разработки
        import warnings

        warnings.filterwarnings(
            "ignore", message="Api key is used with an insecure connection"
        )

        self.client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        self.collection_name = settings.collection_name
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Проверка и создание коллекции если не существует"""
        try:
            collections = self.client.get_collections()
            collection_names = [c.name for c in collections.collections]

            if self.collection_name not in collection_names:
                logger.info(f"Создание коллекции: {self.collection_name}")
                # Получаем размерность модели эмбеддингов
                from app.rag.embedding import get_embedding_model

                embedding_model = get_embedding_model()
                dimension = embedding_model.get_dimension()

                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=dimension, distance=Distance.COSINE
                    ),
                )
                logger.info(f"Коллекция {self.collection_name} создана")
            else:
                logger.info(f"Коллекция {self.collection_name} уже существует")

        except Exception as e:
            logger.error(f"Ошибка при создании коллекции: {e}")
            raise

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """Добавление текстов в векторную базу"""
        try:
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in texts]

            if metadatas is None:
                metadatas = [{} for _ in texts]

            # Создаем эмбеддинги для текстов если не переданы
            if embeddings is None:
                from app.rag.embedding import get_embedding_model
                embedding_model = get_embedding_model()
                embeddings = embedding_model.encode(texts)

            # Создаем точки для Qdrant
            points = []
            for i, (text, metadata, point_id, embedding) in enumerate(
                zip(texts, metadatas, ids, embeddings)
            ):
                point = PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={"text": text, "metadata": metadata},
                )
                points.append(point)

            # Добавляем точки в коллекцию
            self.client.upsert(collection_name=self.collection_name, points=points)

            logger.info(f"Добавлено {len(points)} точек в коллекцию")
            return ids

        except Exception as e:
            logger.error(f"Ошибка при добавлении текстов: {e}")
            raise

    def similarity_search(
        self,
        query: List[float],
        k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Поиск похожих документов"""
        try:
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
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query,  # Ожидается список float
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

            logger.debug(f"Найдено {len(results)} похожих документов")
            return results

        except Exception as e:
            logger.error(f"Ошибка при поиске: {e}")
            raise

    def delete_texts(self, ids: List[str]) -> bool:
        """Удаление текстов по ID"""
        try:
            self.client.delete(
                collection_name=self.collection_name, points_selector=ids
            )
            logger.info(f"Удалено {len(ids)} точек из коллекции")
            return True

        except Exception as e:
            logger.error(f"Ошибка при удалении текстов: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Получение информации о коллекции"""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
            }
        except Exception as e:
            logger.error(f"Ошибка при получении информации о коллекции: {e}")
            raise

    def clear_collection(self) -> bool:
        """Очистка коллекции"""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection_exists()
            logger.info(f"Коллекция {self.collection_name} очищена")
            return True
        except Exception as e:
            logger.error(f"Ошибка при очистке коллекции: {e}")
            raise


def get_vector_store() -> QdrantVectorStore:
    return QdrantVectorStore()
