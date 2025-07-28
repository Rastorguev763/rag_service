# RAG Service

RAG (Retrieval-Augmented Generation) сервис с использованием FastAPI, Qdrant и OpenRouter для работы с русскими текстами.

## 🚀 Возможности

- **RAG система** с векторным поиском на основе Qdrant
- **Русские эмбеддинги** с использованием модели ai-forever/FRIDA
- **LLM интеграция** через OpenRouter (Claude, GPT и др.)
- **Аутентификация** с JWT токенами
- **Управление документами** с загрузкой файлов
- **Чат с контекстом** и историей сообщений
- **REST API** с автоматической документацией

## 🛠 Технологический стек

- **Backend**: FastAPI (Python 3.11+)
- **База данных**: PostgreSQL (пользователи, документы)
- **Векторная БД**: Qdrant
- **LLM**: OpenRouter (Claude, GPT, и др.)
- **Эмбеддинги**: ai-forever/FRIDA
- **Валидация**: Pydantic
- **Логирование**: Loguru
- **Управление пакетами**: Poetry

## 🤖 Модель эмбеддингов ai-forever/FRIDA

**ai-forever/FRIDA** - это специализированная модель для создания эмбеддингов русских текстов, разработанная командой AI Forever.

### Особенности

- **Оптимизирована для русского языка** - лучшие результаты на русскоязычных текстах
- **Высокое качество эмбеддингов** - точный семантический поиск
- **Поддержка контекста** - понимание смысла и контекста текста
- **Стабильность** - проверенная модель в продакшене

### Требования к системе

- **Минимум 8GB RAM** для загрузки модели
- **Рекомендуется 16GB+ RAM** для оптимальной производительности
- **SSD диск** для быстрой загрузки модели

## 📋 Требования

### Минимальные требования

- Python 3.11+
- PostgreSQL
- Qdrant
- OpenRouter API ключ
- **Минимум 8GB RAM** (для модели ai-forever/FRIDA)
- **Рекомендуется 16GB+ RAM** для стабильной работы

### Для Docker развертывания

- Docker Engine 20.10+
- Docker Compose 2.0+
- **Минимум 4GB RAM** для контейнеров
- **Рекомендуется 8GB+ RAM** для продакшена

## 🔧 Установка

### 1. Клонирование репозитория

```bash
git clone <repository-url>
cd rag-service
```

### 2. Установка зависимостей

```bash
# Установка Poetry (если не установлен)
curl -sSL https://install.python-poetry.org | python3 -

# Установка зависимостей
poetry install
```

### 3. Настройка окружения

Скопируйте файл с переменными окружения:

```bash
cp env.example .env
```

Отредактируйте `.env` файл:

```env
# Database settings
DATABASE_URL=postgresql://user:password@localhost:5432/rag_service
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key

# OpenRouter settings
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# JWT settings
SECRET_KEY=your_secret_key_here_make_it_long_and_random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# App settings
DEBUG=True
HOST=0.0.0.0
PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/rag_service.log

# RAG settings
EMBEDDING_MODEL=ai-forever/FRIDA
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
COLLECTION_NAME=documents
```

### 4. Настройка базы данных

```bash
# Создание базы данных PostgreSQL
createdb rag_service

# Применение миграций (если используете Alembic)
alembic upgrade head
```

### 5. Запуск Qdrant

```bash
# Через Docker (рекомендуется)
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# Или установите локально
# https://qdrant.tech/documentation/guides/installation/
```

### 6. Docker развертывание (альтернатива)

Если у вас установлен Docker, можно использовать готовые конфигурации:

```bash
# Клонирование и настройка
git clone <repository-url>
cd rag-service

# Создание .env файла
cp env.example .env
# Отредактируйте .env файл

# Запуск через Docker
docker-compose --profile dev up -d
```

## 🚀 Запуск

### Разработка

#### Локальный запуск

```bash
# Активация виртуального окружения
poetry shell

# Запуск сервера
python -m app.main
```

#### Docker (рекомендуется)

```bash
# Запуск всех сервисов для разработки
docker compose -f docker/docker-compose.dev.yml up -d

# Просмотр логов
docker compose -f docker/docker-compose.dev.yml logs -f rag_service_dev

# Остановка сервисов
docker compose -f docker/docker-compose.dev.yml down
```

### Продакшн

#### Docker Compose

```bash
# Запуск продакшн версии
docker compose -f docker/docker-compose.prod.yml up -d

# Запуск с мониторингом
docker compose -f docker/docker-compose.prod.yml -f docker/docker-compose.monitoring.yml up -d

# Применение миграций вручную (если нужно)
docker compose -f docker/docker-compose.prod.yml exec rag_service_prod alembic upgrade head
```

#### Ручной запуск

```bash
# Через uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Через gunicorn
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📚 API Endpoints

### Аутентификация

- `POST /auth/register` - Регистрация пользователя
- `POST /auth/token` - Получение JWT токена
- `GET /auth/me` - Информация о текущем пользователе

### Чат

- `POST /chat/chat` - Основной endpoint для чата с RAG
- `GET /chat/sessions` - Список сессий чата
- `GET /chat/sessions/{session_id}/messages` - Сообщения сессии
- `DELETE /chat/sessions/{session_id}` - Удаление сессии

### Загрузка документов

- `POST /upload/text` - Загрузка текстового документа
- `POST /upload/file` - Загрузка файла
- `GET /upload/documents` - Список документов пользователя
- `GET /upload/documents/{document_id}` - Получение документа
- `DELETE /upload/documents/{document_id}` - Удаление документа

### Статус системы

- `GET /status/status` - Статус RAG системы
- `GET /status/health` - Проверка здоровья системы
- `GET /status/models` - Список доступных LLM моделей

## 🔐 Безопасность

⚠️ **ВАЖНО**: Проверьте наличие API ключей и паролей в открытом виде в файлах. Не используйте ключи в открытом виде в продакшене.

- Все API ключи хранятся в `.env` файле
- Пароли хешируются с помощью bcrypt
- JWT токены для аутентификации
- CORS настроен для безопасности

## 📖 Примеры использования

### 1. Регистрация и аутентификация

```bash
# Регистрация
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "password123"
  }'

# Получение токена
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=password123"
```

### 2. Загрузка документа

```bash
# Загрузка текста
curl -X POST "http://localhost:8000/upload/text" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Тестовый документ",
    "content": "Это тестовый документ для проверки RAG системы..."
  }'
```

### 3. Чат с RAG

```bash
# Отправка сообщения
curl -X POST "http://localhost:8000/chat/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Расскажи о RAG системе",
    "use_rag": true
  }'
```

## 🧪 Тестирование

```bash
# Запуск тестов
poetry run pytest

# Запуск с покрытием
poetry run pytest --cov=app
```

## 📊 Мониторинг

### Базовый мониторинг

- Логи сохраняются в `logs/rag_service.log`
- Автоматическая ротация логов (10MB, 7 дней)
- Health check endpoint для мониторинга

### Продвинутый мониторинг (Docker)

```bash
# Запуск с мониторингом
docker compose -f docker/docker-compose.prod.yml -f docker/docker-compose.monitoring.yml up -d
```

#### Доступные сервисы мониторинга

- **Prometheus** (порт 9090) - сбор метрик
- **Grafana** (порт 3000) - визуализация дашбордов
- **Node Exporter** (порт 9100) - системные метрики
- **cAdvisor** (порт 8080) - метрики контейнеров
- **Alertmanager** (порт 9093) - уведомления

#### Доступ к интерфейсам

- Grafana: <http://localhost:3000> (admin/admin123)
- Prometheus: <http://localhost:9090>
- Qdrant UI: <http://localhost:6334> (в dev режиме)

## 🔧 Конфигурация

### Основные настройки

Основные настройки в `app/config.py`:

- Размер чанков: `CHUNK_SIZE=1000`
- Перекрытие чанков: `CHUNK_OVERLAP=200`
- Модель эмбеддингов: `EMBEDDING_MODEL=ai-forever/FRIDA`
- LLM модель: `anthropic/claude-3-sonnet:20240229`

### Docker конфигурации

Проект включает несколько Docker Compose конфигураций:

- **`docker-compose.dev.yml`** - для разработки с дополнительными сервисами
- **`docker-compose.prod.yml`** - для продакшена с масштабированием
- **`docker-compose.monitoring.yml`** - для мониторинга и метрик

### Переменные окружения

Создайте файл `.env` в корне проекта:

```env
# Database settings
DATABASE_URL=postgresql://rag_user:rag_password@localhost:5432/rag_service
QDRANT_URL=http://localhost:6333

# OpenRouter settings
OPENROUTER_API_KEY=your_openrouter_api_key

# JWT settings
SECRET_KEY=your_secret_key_here_make_it_long_and_random

# App settings
DEBUG=True
HOST=0.0.0.0
PORT=8000

# RAG settings
EMBEDDING_MODEL=ai-forever/FRIDA
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch
3. Внесите изменения
4. Добавьте тесты
5. Создайте Pull Request

## 📄 Лицензия

MIT License

## 🆘 Поддержка

При возникновении проблем:

1. Проверьте логи в `logs/rag_service.log`
2. Убедитесь, что все сервисы запущены (PostgreSQL, Qdrant)
3. Проверьте правильность API ключей в `.env`
4. Создайте issue в репозитории

### Docker команды для диагностики

```bash
# Проверка статуса контейнеров
docker compose -f docker/docker-compose.dev.yml ps

# Просмотр логов конкретного сервиса
docker compose -f docker/docker-compose.dev.yml logs -f rag_service_dev

# Подключение к базе данных
docker compose -f docker/docker-compose.dev.yml exec postgres psql -U rag_user -d rag_service

# Проверка здоровья сервисов
docker compose -f docker/docker-compose.dev.yml exec rag_service_dev curl http://localhost:8000/status/health

# Пересборка образов
docker compose -f docker/docker-compose.dev.yml build --no-cache
```

### 🔧 Решение проблем с памятью

Если возникает ошибка **"Файл подкачки слишком мал для завершения операции"**:

1. **Увеличьте файл подкачки Windows**:
   - Свойства системы → Дополнительные параметры → Быстродействие → Параметры → Дополнительно → Виртуальная память
   - Установите размер файла подкачки в 1.5-2 раза больше объема RAM

2. **Используйте чат без RAG**:

   ```json
   {
     "message": "ваш вопрос",
     "use_rag": false
   }
   ```

3. **Проверьте доступную память**:
   - Убедитесь, что у вас минимум 8GB свободной RAM
   - Закройте другие приложения, потребляющие много памяти

4. **Альтернативные модели** (если проблема не решается):
   - Измените `EMBEDDING_MODEL` в `.env` на `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

## 🔄 Обновления

### Обновление зависимостей

```bash
# Poetry
poetry update

# Docker
docker compose -f docker/docker-compose.dev.yml build --no-cache
docker compose -f docker/docker-compose.dev.yml up -d
```

### Обновление базы данных

```bash
# Локально
alembic upgrade head

# Docker
docker compose -f docker/docker-compose.dev.yml exec rag_service_dev alembic upgrade head
```

### Обновление Docker образов

```bash
# Обновление всех образов
docker compose -f docker/docker-compose.dev.yml pull
docker compose -f docker/docker-compose.dev.yml up -d

# Обновление конкретного сервиса
docker compose -f docker/docker-compose.dev.yml pull rag_service_dev
docker compose -f docker/docker-compose.dev.yml up -d rag_service_dev
```
