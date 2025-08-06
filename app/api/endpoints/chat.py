import traceback
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_async_db
from app.models.models import ChatMessage, ChatSession, User
from app.rag.llm import OpenRouterLLM, get_llm
from app.rag.rag_service import RAGService, get_rag_service
from app.schemas.schemas import ChatMessage as ChatMessageSchema
from app.schemas.schemas import ChatRequest, ChatResponse
from app.schemas.schemas import ChatSession as ChatSessionSchema
from app.utils.auth import get_current_active_user
from app.utils.logger import logger

router = APIRouter()


@router.post("", response_model=ChatResponse)
async def chat_with_rag(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
    llm: OpenRouterLLM = Depends(get_llm),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Основной endpoint для чата с RAG"""
    try:

        # Получаем или создаем сессию чата
        if request.session_id:
            result = await db.execute(
                select(ChatSession).filter(
                    ChatSession.id == request.session_id,
                    ChatSession.user_id == current_user.id,
                )
            )
            session = result.scalar_one_or_none()
            if not session:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Chat session not found",
                )
        else:
            # Создаем новую сессию
            session = ChatSession(
                title=(
                    request.message[:50] + "..."
                    if len(request.message) > 50
                    else request.message
                ),
                user_id=current_user.id,
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)

        # Сохраняем сообщение пользователя в БД
        user_message = ChatMessage(
            content=request.message, role="user", session_id=session.id
        )
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        # Сохраняем сообщение пользователя в персональную коллекцию Qdrant
        try:
            message_metadata = {
                "session_id": session.id,
                "message_id": user_message.id,
                "role": "user",
                "content_type": "chat_message",
            }
            await rag_service.add_user_message(
                user_id=current_user.id,
                message_text=request.message,
                metadata=message_metadata,
            )
            logger.info(
                f"Сообщение пользователя {current_user.id} сохранено в персональную коллекцию"
            )
        except Exception as e:
            logger.error(f"Ошибка при сохранении сообщения в коллекцию: {e}")
            # Не прерываем чат, если не удалось сохранить в коллекцию

        # Получаем контекст если используется RAG
        context = None
        sources = None
        k_points_used = 0
        if request.use_rag:
            # Используем гибридный поиск по документам и сообщениям пользователя
            similar_docs = await rag_service.search_hybrid(
                query=request.message,
                k=request.k_points,
                user_id=current_user.id,
                user_messages_weight=0.3,  # 30% веса для сообщений пользователя
                documents_weight=0.7,  # 70% веса для документов
            )

            if similar_docs:
                logger.info(
                    f"Найдены похожие документы и сообщения: {len(similar_docs)}"
                )
                context_parts = []
                sources = []
                for doc_text, score, metadata in similar_docs:
                    content_type = metadata.get("content_type", "unknown")
                    if content_type == "document_chunk":
                        source_name = metadata.get("document_title", "Unknown Document")
                    elif content_type == "chat_message":
                        source_name = f"Previous Message (Session {metadata.get('session_id', 'Unknown')})"
                    else:
                        source_name = "Unknown Source"

                    logger.info(
                        f"Источник: {source_name}, тип: {content_type}, релевантность: {score}"
                    )
                    if score > 0.6:  # Снижаем порог релевантности для гибридного поиска
                        context_parts.append(doc_text)
                        sources.append(source_name)
                        k_points_used += 1

                if context_parts:
                    context = "\n\n".join(context_parts)
                    logger.info(
                        f"Сформирован контекст для LLM ({len(context)} символов):"
                    )
                    context_for_logging = (
                        context[:500] + "..." if len(context) > 500 else context
                    )
                    logger.info(f"Контекст:\n{context_for_logging}")
                else:
                    logger.info("Контекст не сформирован - нет релевантных документов")
        else:
            logger.info("RAG не используется")

        # Получаем историю сообщений для контекста
        conversation_history = []

        # Явно загружаем сообщения сессии
        await db.refresh(session, attribute_names=["messages"])

        if session.messages:
            for msg in session.messages[-10:]:  # Последние 10 сообщений
                conversation_history.append({"role": msg.role, "content": msg.content})

            logger.info(
                f"Загружена история сообщений: {len(conversation_history)} сообщений"
            )
        else:
            logger.info("История сообщений пуста")

        # Генерируем ответ
        response_text = await llm.chat_completion(
            user_message=request.message,
            conversation_history=conversation_history,
            use_rag=request.use_rag,
            context=context,
            max_tokens=request.max_tokens,
        )

        # Сохраняем ответ ассистента в БД
        assistant_message = ChatMessage(
            content=response_text, role="assistant", session_id=session.id
        )
        db.add(assistant_message)
        await db.commit()
        await db.refresh(assistant_message)

        # Сохраняем ответ ассистента в персональную коллекцию пользователя
        try:
            assistant_metadata = {
                "session_id": session.id,
                "message_id": assistant_message.id,
                "role": "assistant",
                "content_type": "chat_message",
            }
            await rag_service.add_user_message(
                user_id=current_user.id,
                message_text=response_text,
                metadata=assistant_metadata,
            )
            logger.info(
                f"Ответ ассистента для пользователя {current_user.id} сохранен в персональную коллекцию"
            )
        except Exception as e:
            logger.error(f"Ошибка при сохранении ответа ассистента в коллекцию: {e}")

        logger.info(
            f"Сгенерирован ответ для пользователя {current_user.username} (использовано {k_points_used} точек контекста)"
        )

        return ChatResponse(
            message=response_text,
            session_id=session.id,
            message_id=assistant_message.id,
            sources=sources,
            k_points_used=k_points_used if request.use_rag else None,
        )

    except Exception as e:
        logger.error(f"Ошибка в чате: {e}")
        logger.error(traceback.format_exc())
        raise


@router.get("/sessions", response_model=List[ChatSessionSchema])
async def get_chat_sessions(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Получение списка сессий чата пользователя"""
    try:
        result = await db.execute(
            select(ChatSession).filter(ChatSession.user_id == current_user.id)
        )
        sessions = result.scalars().all()

        # Явно загружаем связанные данные для каждого объекта
        for session in sessions:
            await db.refresh(session, attribute_names=["messages"])

        return sessions

    except Exception as e:
        logger.error(f"Ошибка при получении сессий: {e}")
        raise


@router.get("/sessions/{session_id}/messages", response_model=List[ChatMessageSchema])
async def get_chat_messages(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Получение сообщений конкретной сессии"""
    try:
        # Проверяем права доступа
        result = await db.execute(
            select(ChatSession).filter(
                ChatSession.id == session_id, ChatSession.user_id == current_user.id
            )
        )
        session = result.scalar_one_or_none()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
            )

        result = await db.execute(
            select(ChatMessage).filter(ChatMessage.session_id == session_id)
        )
        messages = result.scalars().all()

        # Явно загружаем связанные данные для каждого сообщения
        for message in messages:
            await db.refresh(message, attribute_names=["session"])

        return messages

    except Exception as e:
        logger.error(f"Ошибка при получении сообщений: {e}")
        raise


@router.delete("/sessions/{session_id}")
async def delete_chat_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Удаление сессии чата"""
    try:
        result = await db.execute(
            select(ChatSession).filter(
                ChatSession.id == session_id, ChatSession.user_id == current_user.id
            )
        )
        session = result.scalar_one_or_none()

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Chat session not found"
            )

        # Удаляем все сообщения сессии

        await db.execute(
            delete(ChatMessage).filter(ChatMessage.session_id == session_id)
        )

        # Удаляем сессию
        await db.delete(session)
        await db.commit()

        logger.info(f"Удалена сессия чата {session_id}")
        return {"message": "Chat session deleted"}

    except Exception as e:
        logger.error(f"Ошибка при удалении сессии: {e}")
        await db.rollback()
        raise


@router.get("/collection-info")
async def get_user_collection_info(
    current_user: User = Depends(get_current_active_user),
):
    """Получение информации о персональной коллекции пользователя"""
    try:
        rag_service = get_rag_service()
        collection_info = await rag_service.get_user_collection_info(current_user.id)
        return collection_info
    except Exception as e:
        logger.error(f"Ошибка при получении информации о коллекции: {e}")
        raise
