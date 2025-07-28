import traceback
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.utils.auth import get_current_active_user
from app.db.database import get_async_db
from app.utils.logger import logger
from app.models.models import ChatMessage, ChatSession, User
from app.rag.llm import get_llm
from app.rag.rag_service import get_rag_service
from app.schemas.schemas import ChatMessage as ChatMessageSchema
from app.schemas.schemas import ChatRequest, ChatResponse
from app.schemas.schemas import ChatSession as ChatSessionSchema

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat_with_rag(
    request: ChatRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Основной endpoint для чата с RAG"""
    try:
        llm = get_llm()

        # Инициализируем RAG сервис только если он нужен
        rag_service = None
        if request.use_rag:
            rag_service = get_rag_service()

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

        # Сохраняем сообщение пользователя
        user_message = ChatMessage(
            content=request.message, role="user", session_id=session.id
        )
        db.add(user_message)
        await db.commit()
        await db.refresh(user_message)

        # Получаем контекст если используется RAG
        context = None
        sources = None
        if request.use_rag and rag_service:
            # Ищем похожие документы
            similar_docs = rag_service.search_similar(
                query=request.message, k=3, user_id=current_user.id
            )

            if similar_docs:
                context_parts = []
                sources = []
                for doc_text, score, metadata in similar_docs:
                    if score > 0.7:  # Порог релевантности
                        context_parts.append(doc_text)
                        sources.append(metadata.get("document_title", "Unknown"))

                if context_parts:
                    context = "\n\n".join(context_parts)
        else:
            logger.info(f"RAG не используется {request.use_rag=} - {rag_service=}")

        # Получаем историю сообщений для контекста
        conversation_history = []
        if session.messages:
            for msg in session.messages[-10:]:  # Последние 10 сообщений
                conversation_history.append({"role": msg.role, "content": msg.content})

        # Генерируем ответ
        response_text = llm.chat_completion(
            user_message=request.message,
            conversation_history=conversation_history,
            use_rag=request.use_rag,
            context=context,
            max_tokens=request.max_tokens,
        )

        # Сохраняем ответ ассистента
        assistant_message = ChatMessage(
            content=response_text, role="assistant", session_id=session.id
        )
        db.add(assistant_message)
        await db.commit()
        await db.refresh(assistant_message)

        logger.info(f"Сгенерирован ответ для пользователя {current_user.username}")

        return ChatResponse(
            message=response_text,
            session_id=session.id,
            message_id=assistant_message.id,
            sources=sources,
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
        from sqlalchemy import delete

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
