import os
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.utils.auth import get_current_active_user
from app.db.database import get_async_db
from app.utils.logger import logger
from app.models.models import User, Document
from app.rag.rag_service import RAGService, get_rag_service
from app.schemas.schemas import Document as DocumentSchema
from app.schemas.schemas import DocumentCreate, UploadResponse

router = APIRouter()


@router.post("/text", response_model=UploadResponse)
async def upload_text_document(
    document_data: DocumentCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Загрузка текстового документа в RAG"""
    try:
        # Обрабатываем документ
        document = await rag_service.process_document(
            db=db, document_data=document_data, user=current_user
        )

        logger.info(
            f"Документ {document.id} загружен пользователем {current_user.username}"
        )

        return UploadResponse(
            document_id=document.id,
            title=document.title,
            status="processed",
            message="Document successfully processed and added to RAG",
        )

    except Exception as e:
        logger.error(f"Ошибка при загрузке документа: {e}")
        raise


@router.post("/file", response_model=UploadResponse)
async def upload_file_document(
    file: UploadFile = File(...),
    title: str = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
    rag_service: RAGService = Depends(get_rag_service),
):
    """Загрузка файла в RAG"""
    try:
        # Проверяем тип файла
        allowed_extensions = [".txt", ".md", ".pdf", ".docx"]
        file_extension = os.path.splitext(file.filename)[1].lower()

        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_extension} not supported. Allowed: {allowed_extensions}",
            )

        # Читаем содержимое файла
        content = await file.read()

        # Для текстовых файлов декодируем
        if file_extension in [".txt", ".md"]:
            try:
                text_content = content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = content.decode("latin-1")
        else:
            # Для других форматов пока возвращаем ошибку
            # TODO: Добавить обработку PDF/DOCX
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PDF and DOCX processing not implemented yet",
            )

        # Создаем объект документа
        document_data = DocumentCreate(
            title=title or file.filename,
            content=text_content,
            file_path=file.filename,
            file_type=file_extension,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        # Обрабатываем документ
        document = await rag_service.process_document(
            db=db, document_data=document_data, user=current_user
        )

        logger.info(
            f"Файл {file.filename} загружен пользователем {current_user.username}"
        )

        return UploadResponse(
            document_id=document.id,
            title=document.title,
            status="processed",
            message="File successfully processed and added to RAG",
        )

    except Exception as e:
        logger.error(f"Ошибка при загрузке файла: {e}")
        raise


@router.get("/documents", response_model=List[DocumentSchema])
async def get_user_documents(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Получение списка документов пользователя"""
    try:
        result = await db.execute(
            select(Document).filter(Document.owner_id == current_user.id)
        )
        documents = result.scalars().all()

        # Явно загружаем связанные данные для каждого документа
        for document in documents:
            await db.refresh(document, attribute_names=["owner", "chunks"])

        return documents

    except Exception as e:
        logger.error(f"Ошибка при получении документов: {e}")
        raise


@router.get("/documents/{document_id}", response_model=DocumentSchema)
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Получение конкретного документа"""
    try:
        result = await db.execute(
            select(Document).filter(
                Document.id == document_id, Document.owner_id == current_user.id
            )
        )
        document = result.scalar_one_or_none()

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        # Явно загружаем связанные данные
        await db.refresh(document, attribute_names=["owner", "chunks"])

        return document

    except Exception as e:
        logger.error(f"Ошибка при получении документа: {e}")
        raise


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_async_db),
):
    """Удаление документа"""
    try:
        rag_service = get_rag_service()

        success = await rag_service.delete_document(
            db=db, document_id=document_id, user=current_user
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
            )

        logger.info(
            f"Документ {document_id} удален пользователем {current_user.username}"
        )
        return {"message": "Document deleted successfully"}

    except Exception as e:
        logger.error(f"Ошибка при удалении документа: {e}")
        raise
