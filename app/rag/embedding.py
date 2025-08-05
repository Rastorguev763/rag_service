import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.config.config import settings
from app.utils.logger import logger


class EmbeddingModel:
    """Модель для создания эмбеддингов русских текстов"""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Загрузка модели эмбеддингов: {self.model_name}")
        logger.info(f"Устройство: {self.device}")
        self.model = SentenceTransformer(self.model_name).to(self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Модель загружена, размерность: {self.dimension}")

        # Создаём пул потоков (один экземпляр на модель)
        self.executor = ThreadPoolExecutor(max_workers=2)  # Лучше не больше 1-2 для GPU

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Создание эмбеддингов для текста или списка текстов (синхронный)"""
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

    async def encode_async(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Асинхронная обёртка для encode. Не блокирует event loop."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, self.encode, texts  # передаём синхронный метод
        )

    def encode_single(self, text: str) -> List[float]:
        """Создание эмбеддинга для одного текста"""
        embedding = self.encode(text)
        return embedding[0].tolist()

    async def encode_single_async(self, text: str) -> List[float]:
        """Асинхронная версия для одного текста"""
        embedding = await self.encode_async(text)
        return embedding[0].tolist()

    def get_dimension(self) -> int:
        """Получение размерности эмбеддингов"""
        return self.dimension

    def close(self):
        """Закрытие пула потоков (вызывать при завершении приложения)"""
        self.executor.shutdown(wait=True)


# Глобальный экземпляр
embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    global embedding_model
    if embedding_model is None:
        embedding_model = EmbeddingModel()
    return embedding_model
