# Используем Python 3.12 slim образ
FROM python:3.12-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Обновляем pip
RUN pip install --upgrade pip

# Устанавливаем Poetry
RUN pip install poetry

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock* ./

# Устанавливаем Python зависимости через Poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main --no-interaction --no-ansi

# Копируем исходный код
COPY . .

# Создаем директорию для логов
RUN mkdir -p logs

# Копируем скрипт запуска
COPY docker/start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Создаем пользователя для безопасности
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Открываем порт
EXPOSE 8000

# Команда по умолчанию
CMD ["/app/start.sh", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]