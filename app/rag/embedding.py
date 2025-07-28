from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config.config import settings
from app.utils.logger import logger


class EmbeddingModel:
    """Модель для создания эмбеддингов русских текстов"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Модель загружена, размерность: {self.dimension}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Создание эмбеддингов для текста или списка текстов"""
        try:
            if isinstance(texts, str):
                texts = [texts]

            embeddings = self.model.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True
            )

            logger.debug(f"Создано эмбеддингов: {len(embeddings)}")
            return embeddings

        except Exception as e:
            logger.error(f"Ошибка при создании эмбеддингов: {e}")
            raise

    def encode_single(self, text: str) -> List[float]:
        """Создание эмбеддинга для одного текста"""
        embedding = self.encode(text)
        return embedding[0].tolist()

    def get_dimension(self) -> int:
        """Получение размерности эмбеддингов"""
        return self.dimension


embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    global embedding_model
    if embedding_model is None:
        embedding_model = EmbeddingModel()
    return embedding_model
