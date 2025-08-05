from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, model_validator, Field
from app.config.config import settings


# User schemas
class UserBase(BaseModel):
    username: str
    email: EmailStr


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None


class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# Document schemas
class DocumentBase(BaseModel):
    title: str
    content: str
    file_path: Optional[str] = None
    file_type: Optional[str] = None


class DocumentCreate(DocumentBase):
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200


class DocumentUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    is_processed: Optional[bool] = None


class Document(DocumentBase):
    id: int
    owner_id: int
    chunk_size: int
    chunk_overlap: int
    is_processed: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# Chat schemas
class ChatMessageBase(BaseModel):
    content: str
    role: str


class ChatMessageCreate(ChatMessageBase):
    pass


class ChatMessage(ChatMessageBase):
    id: int
    session_id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatSessionBase(BaseModel):
    title: Optional[str] = None


class ChatSessionCreate(ChatSessionBase):
    pass


class ChatSession(ChatSessionBase):
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    messages: List[ChatMessage] = []

    model_config = ConfigDict(from_attributes=True)


# Chat request/response schemas
class ChatRequest(BaseModel):
    message: str = Field(
        ..., min_length=1, max_length=10000, description="Текст сообщения пользователя"
    )
    session_id: Optional[int] = None
    use_rag: bool = True
    max_tokens: Optional[int] = Field(default=1000, ge=1, le=4000)
    k_points: Optional[int] = Field(
        default=settings.default_k_points,
        ge=1,
        le=20,
        description="Количество точек (чанков) для формирования контекста (1-20)",
    )

    @model_validator(mode="after")
    def validate_parameters(self) -> "ChatRequest":
        """Валидация параметров запроса"""
        # Валидация k_points
        if self.k_points is not None:
            if self.k_points < 1:
                self.k_points = 1
            elif self.k_points > 20:
                self.k_points = 20

        # Валидация max_tokens
        if self.max_tokens is not None:
            if self.max_tokens < 1:
                self.max_tokens = 1
            elif self.max_tokens > 4000:
                self.max_tokens = 4000

        return self


class ChatResponse(BaseModel):
    message: str
    session_id: int
    message_id: int
    sources: Optional[List[str]] = None
    k_points_used: Optional[int] = None


# Upload schemas
class UploadResponse(BaseModel):
    document_id: int
    title: str
    status: str
    message: str


# Status schemas
class RAGStatus(BaseModel):
    total_documents: int
    processed_documents: int
    total_chunks: int
    collection_size: int
    is_healthy: bool
    last_update: datetime


# Auth schemas
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None
