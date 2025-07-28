from datetime import UTC, datetime
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.utils.auth import get_current_active_user
from app.db.database import get_async_db
from app.utils.logger import logger
from app.models.models import User
from app.rag.rag_service import RAGService, get_rag_service
from app.schemas.schemas import RAGStatus

router = APIRouter()


@router.get("/status", response_model=RAGStatus)
async def get_rag_status(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Получение статуса RAG системы"""
    try:
        status_info = await rag_service.get_rag_status(db)

        logger.info(f"Статус RAG запрошен пользователем {current_user.username}")
        return status_info

    except Exception as e:
        logger.error(f"Ошибка при получении статуса RAG: {e}")
        raise


@router.get("/health")
async def health_check(rag_service: RAGService = Depends(get_rag_service)):
    """Проверка здоровья системы"""
    try:
        # Проверяем подключение к Qdrant
        try:
            rag_service.vector_store.get_collection_info()
            qdrant_healthy = True
        except Exception:
            qdrant_healthy = False

        # Проверяем модель эмбеддингов
        try:
            _ = rag_service.embedding_model.get_dimension()
            embedding_healthy = True
        except Exception:
            embedding_healthy = False

        overall_health = qdrant_healthy and embedding_healthy

        return {
            "status": "healthy" if overall_health else "unhealthy",
            "components": {
                "qdrant": "healthy" if qdrant_healthy else "unhealthy",
                "embedding_model": "healthy" if embedding_healthy else "unhealthy",
            },
            "timestamp": datetime.now(UTC).isoformat(),
        }

    except Exception as e:
        logger.error(f"Ошибка при проверке здоровья: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now(UTC).isoformat(),
        }
