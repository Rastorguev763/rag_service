from typing import Dict, List, Optional

from openai import AsyncOpenAI

from app.config.config import settings
from app.utils.logger import logger


class OpenRouterLLM:
    """Класс для работы с LLM через OpenRouter (асинхронная версия)"""

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url
        )
        self.default_model = "deepseek/deepseek-r1-0528:free"
        logger.info("Инициализирован асинхронный OpenRouter клиент")

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 5000,
        temperature: float = 0.7,
        use_rag: bool = True,
        context: Optional[str] = None,
    ) -> str:
        """Генерация ответа от LLM (асинхронно)"""
        try:
            # Формируем промпт с контекстом если используется RAG
            if use_rag and context:
                system_message = {
                    "role": "system",
                    "content": (
                        f"""
                        Ты полезный ассистент. Используй следующую информацию для ответа на вопрос пользователя:

                        Контекст:
                        {context}

                        Отвечай на основе предоставленного контекста. Если в контексте нет информации для ответа, скажи об этом честно.
                        """
                    ),
                }
                messages = [system_message] + messages

            # Логируем промт для отладки
            prompt_for_logging = self._format_messages_for_logging(messages)
            logger.info(
                f"Отправляем промт в LLM (модель: {model or self.default_model}):"
            )
            logger.info(f"Промт:\n{prompt_for_logging}")
            logger.info(
                f"Параметры: max_tokens={max_tokens}, temperature={temperature}"
            )
            # Выполняем асинхронный запрос к LLM
            response = await self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            answer = response.choices[0].message.content

            # Логируем ответ от LLM
            answer_for_logging = answer[:500] + "..." if len(answer) > 500 else answer
            logger.info(f"Получен ответ от LLM ({len(answer)} символов):")
            logger.info(f"Ответ:\n{answer_for_logging}")

            return answer

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            raise

    def _format_messages_for_logging(self, messages: List[Dict[str, str]]) -> str:
        """Форматирует сообщения для логирования"""
        formatted_prompt = ""
        for i, message in enumerate(messages):
            role = message.get("role", "unknown")
            content = message.get("content", "")

            # Обрезаем длинный контент для читаемости логов
            if len(content) > 1000:
                content = content[:1000] + "... [обрезано]"

            formatted_prompt += f"\n--- Сообщение {i+1} ({role}) ---\n"
            formatted_prompt += f"{content}\n"

        return formatted_prompt

    async def chat_completion(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        context: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Удобный метод для чат-комплектации (асинхронно)"""
        messages = conversation_history or []

        # Проверяем, не содержится ли уже текущее сообщение в истории
        # (может быть, если сообщение уже сохранено в БД)
        current_message_in_history = any(
            msg.get("content") == user_message and msg.get("role") == "user"
            for msg in messages
        )

        if not current_message_in_history:
            messages.append({"role": "user", "content": user_message})
            logger.debug("Добавлено текущее сообщение пользователя к истории")
        else:
            logger.debug("Текущее сообщение пользователя уже содержится в истории")

        return await self.generate_response(
            messages=messages, use_rag=use_rag, context=context, **kwargs
        )


def get_llm() -> OpenRouterLLM:
    return OpenRouterLLM()
