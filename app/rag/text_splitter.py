import re
from typing import List

from app.config.config import settings
from app.utils.logger import logger


class TextSplitter:
    """Класс для разбиения текстов на чанки"""

    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        logger.info(
            f"Инициализирован TextSplitter: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}"
        )

    def split_text(self, text: str) -> List[str]:
        """Разбиение текста на чанки"""
        try:
            if not text.strip():
                return []

            # Очищаем текст от лишних пробелов
            text = re.sub(r"\s+", " ", text.strip())

            # Если текст меньше размера чанка, возвращаем как есть
            if len(text) <= self.chunk_size:
                return [text]

            chunks = []
            start = 0

            while start < len(text):
                # Определяем конец чанка
                end = start + self.chunk_size

                # Если это не последний чанк, ищем хорошее место для разрыва
                if end < len(text):
                    # Ищем последний пробел или знак препинания
                    last_space = text.rfind(" ", start, end)
                    last_punct = max(
                        text.rfind(".", start, end),
                        text.rfind("!", start, end),
                        text.rfind("?", start, end),
                        text.rfind("\n", start, end),
                    )

                    # Выбираем лучшее место для разрыва
                    if last_punct > last_space and last_punct > start:
                        end = last_punct + 1
                    elif last_space > start:
                        end = last_space + 1

                # Добавляем чанк
                chunk = text[start:end].strip()
                if chunk:
                    chunks.append(chunk)

                # Переходим к следующему чанку с учетом перекрытия
                start = end - self.chunk_overlap
                if start >= len(text):
                    break

            logger.debug(f"Текст разбит на {len(chunks)} чанков")
            return chunks

        except Exception as e:
            logger.error(f"Ошибка при разбиении текста: {e}")
            raise

    def split_documents(self, documents: List[str]) -> List[str]:
        """Разбиение списка документов на чанки"""
        all_chunks = []
        for i, doc in enumerate(documents):
            chunks = self.split_text(doc)
            all_chunks.extend(chunks)
            logger.debug(f"Документ {i+1} разбит на {len(chunks)} чанков")

        logger.info(
            f"Всего создано {len(all_chunks)} чанков из {len(documents)} документов"
        )
        return all_chunks


def get_text_splitter() -> TextSplitter:
    return TextSplitter()
