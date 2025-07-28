from typing import Dict, List, Optional

from openai import OpenAI

from app.config.config import settings
from app.utils.logger import logger


class OpenRouterLLM:
    """Класс для работы с LLM через OpenRouter"""

    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openrouter_api_key, base_url=settings.openrouter_base_url
        )
        self.default_model = "qwen/qwen3-235b-a22b-2507:free"
        logger.info("Инициализирован OpenRouter клиент")

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        max_tokens: int = 1000,
        temperature: float = 0.7,
        use_rag: bool = True,
        context: Optional[str] = None,
    ) -> str:
        """Генерация ответа от LLM"""
        try:
            # Формируем промпт с контекстом если используется RAG
            if use_rag and context:
                system_message = {
                    "role": "system",
                    "content": f"""Ты полезный ассистент. Используй следующую информацию для ответа на вопрос пользователя:

Контекст:
{context}

Отвечай на основе предоставленного контекста. Если в контексте нет информации для ответа, скажи об этом честно.""",
                }
                messages = [system_message] + messages

            # Выполняем запрос к LLM
            response = self.client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            answer = response.choices[0].message.content
            logger.debug(f"Получен ответ от LLM: {len(answer)} символов")
            return answer

        except Exception as e:
            logger.error(f"Ошибка при генерации ответа: {e}")
            raise

    def chat_completion(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        context: Optional[str] = None,
        **kwargs,
    ) -> str:
        """Удобный метод для чат-комплектации"""
        messages = conversation_history or []
        messages.append({"role": "user", "content": user_message})

        return self.generate_response(
            messages=messages, use_rag=use_rag, context=context, **kwargs
        )


def get_llm() -> OpenRouterLLM:
    """Получение глобального экземпляра LLM"""
    return OpenRouterLLM()
