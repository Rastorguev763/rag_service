from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr


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

    class Config:
        from_attributes = True


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

    class Config:
        from_attributes = True


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

    class Config:
        from_attributes = True


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

    class Config:
        from_attributes = True


# Chat request/response schemas
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[int] = None
    use_rag: bool = True
    max_tokens: Optional[int] = 1000


class ChatResponse(BaseModel):
    message: str
    session_id: int
    message_id: int
    sources: Optional[List[str]] = None


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
