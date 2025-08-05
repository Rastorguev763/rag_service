from typing import Optional

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database settings
    database_url: str = "postgresql://user:password@localhost:5432/rag_service"
    database_url_async: str = (
        "postgresql+asyncpg://user:password@localhost:5432/rag_service"
    )
    postgres_db: str = "rag_service"
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # Database pool settings
    db_echo: bool = False
    db_pool_size: int = 10
    db_max_overflow: int = 20
    db_pool_timeout: int = 30

    # OpenRouter settings
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # JWT settings
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # App settings
    host: str = "0.0.0.0"
    port: int = 8000

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/rag_service.log"

    # RAG settings
    embedding_model: str = "ai-forever/FRIDA"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    collection_name: str = "documents"
    default_k_points: int = 3  # Количество точек по умолчанию для RAG

    model_config = ConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False
    )


settings = Settings()
