import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.endpoints import auth, chat, status, upload
from app.config.config import settings
from app.rag.embedding import get_embedding_model
from app.utils.logger import logger

# Отключаем предупреждение о symlinks для Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan события для FastAPI"""
    # Startup
    logger.info("RAG Service запущен")
    logger.info(f"Уровень логирования: {settings.log_level}")
    logger.info(f"Хост: {settings.host}:{settings.port}")
    embedding_model = get_embedding_model()

    yield

    # Закрываем пул потоков модели эмбеддингов
    embedding_model.close()
    logger.info("Пул потоков модели эмбеддингов закрыт")
    # Shutdown
    logger.info("RAG Service остановлен")


# Создаем FastAPI приложение
app = FastAPI(
    title="RAG Service API",
    description="RAG сервис с FastAPI, Qdrant и OpenRouter",
    version="1.0.0",
    lifespan=lifespan,
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роутеры
app.include_router(auth.router, prefix="/auth", tags=["authentication"])
app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(upload.router, prefix="/upload", tags=["upload"])
app.include_router(status.router, prefix="/status", tags=["status"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=True)
