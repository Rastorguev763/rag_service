import sys
from pathlib import Path

from loguru import logger

from app.config.config import settings

# Создаем директорию для логов
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Удаляем стандартный обработчик
logger.remove()

# Добавляем консольный обработчик
console_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)
logger.add(sys.stdout, format=console_format, level=settings.log_level, colorize=True)

# Добавляем файловый обработчик
file_format = (
    "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | " "{name}:{function}:{line} - {message}"
)
logger.add(
    settings.log_file,
    format=file_format,
    level=settings.log_level,
    rotation="10 MB",
    retention="7 days",
    compression="zip",
)

# Экспортируем настроенный логгер
__all__ = ["logger"]
