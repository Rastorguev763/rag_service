# RAG Service

RAG (Retrieval-Augmented Generation) сервис с использованием FastAPI, Qdrant и OpenRouter для работы с русскими текстами.

## 🚀 Возможности

- **RAG система** с векторным поиском на основе Qdrant
- **Русские эмбеддинги** с использованием модели ai-forever/FRIDA
- **LLM интеграция** через OpenRouter (DeepSeek, Claude, GPT и др.)
- **Аутентификация** с JWT токенами
- **Управление документами** с загрузкой файлов
- **Чат с контекстом** и историей сообщений
- **Настраиваемое количество точек контекста** (k_points) для оптимизации качества ответов
- **REST API** с автоматической документацией
- **Docker Compose** для простого развертывания

## 🛠 Технологический стек

- **Backend**: FastAPI (Python 3.12+)
- **База данных**: PostgreSQL (пользователи, документы)
- **Векторная БД**: Qdrant
- **LLM**: OpenRouter (DeepSeek R1, Claude, GPT, и др.)
- **Эмбеддинги**: ai-forever/FRIDA
- **Валидация**: Pydantic v2
- **Логирование**: Loguru
- **Контейнеризация**: Docker & Docker Compose

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

- Docker Engine 20.10+
- Docker Compose 2.0+
- OpenRouter API ключ
- **Минимум 8GB RAM** (для модели ai-forever/FRIDA)
- **Рекомендуется 16GB+ RAM** для стабильной работы
- **SSD диск** для быстрой работы векторной БД

## 🚀 Быстрый старт

### 1. Клонирование и настройка

```bash
git clone <repository-url>
cd rag-service
cp env.example .env
```

### 2. Настройка API ключей

Отредактируйте `.env` файл и добавьте ваш OpenRouter API ключ:

```env
# OpenRouter settings
OPENROUTER_API_KEY=your_openrouter_api_key

# JWT settings  
SECRET_KEY=your_secret_key_here_make_it_long_and_random
```

### 3. Запуск сервисов

```bash
# Разработка
docker compose -f docker/docker-compose.dev.yml up -d

# Или продакшн
docker compose -f docker/docker-compose.prod.yml up -d
```

### 4. Проверка работы

- API документация: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/status/health>

## 🔧 Подробная установка

### Настройка окружения

Полная конфигурация `.env` файла:

```env
# Database settings
DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_service
DATABASE_URL_ASYNC=postgresql+asyncpg://rag_user:rag_password@postgres:5432/rag_service
POSTGRES_DB=rag_service
POSTGRES_USER=rag_user
POSTGRES_PASSWORD=rag_password

# Qdrant settings
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=your_qdrant_api_key

# OpenRouter settings
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# JWT settings
SECRET_KEY=your_secret_key_here_make_it_long_and_random
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# App settings
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
DEFAULT_K_POINTS=3
```

## 🚀 Запуск

### Разработка

```bash
# Запуск всех сервисов для разработки
docker compose -f docker/docker-compose.dev.yml up -d

# Просмотр логов
docker compose -f docker/docker-compose.dev.yml logs -f rag_service_dev

# Остановка сервисов
docker compose -f docker/docker-compose.dev.yml down
```

### Продакшн

```bash
# Запуск продакшн версии
docker compose -f docker/docker-compose.prod.yml up -d

# Запуск с мониторингом
docker compose -f docker/docker-compose.prod.yml -f docker/docker-compose.monitoring.yml up -d

# Применение миграций вручную (если нужно)
docker compose -f docker/docker-compose.prod.yml exec rag_service_prod alembic upgrade head
```

### Доступ к сервисам

После запуска сервисы будут доступны по следующим адресам:

- **RAG API**: <http://localhost:8000>
- **API документация**: <http://localhost:8000/docs>
- **Qdrant UI** (dev): <http://localhost:6334>
- **Grafana** (мониторинг): <http://localhost:3000>
- **Prometheus** (метрики): <http://localhost:9090>

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
# Отправка сообщения с настройкой количества точек контекста
curl -X POST "http://localhost:8000/chat/chat" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Расскажи о RAG системе",
    "use_rag": true,
    "k_points": 5,
    "max_tokens": 1000
  }'
```

**Параметры чата:**

- `message` - текст сообщения
- `use_rag` - использовать ли RAG (по умолчанию: `true`)
- `k_points` - количество точек (чанков) для формирования контекста (1-20, по умолчанию: 3)
- `max_tokens` - максимальное количество токенов в ответе
- `session_id` - ID сессии чата (опционально)

## 📊 Мониторинг

### Базовый мониторинг

- Логи сохраняются в `logs/rag_service.log`
- Автоматическая ротация логов (10MB, 7 дней)
- Health check endpoint для мониторинга

### Продвинутый мониторинг

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

## 🔧 Конфигурация

### Основные настройки

Основные настройки в `app/config/config.py`:

- Размер чанков: `CHUNK_SIZE=1000`
- Перекрытие чанков: `CHUNK_OVERLAP=200`
- Модель эмбеддингов: `EMBEDDING_MODEL=ai-forever/FRIDA`
- LLM модель: `deepseek/deepseek-r1-0528:free`
- Количество точек по умолчанию: `DEFAULT_K_POINTS=3`

### Docker конфигурации

Проект включает несколько Docker Compose конфигураций:

- **`docker/docker-compose.dev.yml`** - для разработки с дополнительными сервисами
- **`docker/docker-compose.prod.yml`** - для продакшена с масштабированием
- **`docker/docker-compose.monitoring.yml`** - для мониторинга и метрик

### Переменные окружения

Основные переменные в `.env` файле:

```env
# Database settings
DATABASE_URL=postgresql://rag_user:rag_password@postgres:5432/rag_service
QDRANT_URL=http://qdrant:6333

# OpenRouter settings
OPENROUTER_API_KEY=your_openrouter_api_key

# JWT settings
SECRET_KEY=your_secret_key_here_make_it_long_and_random

# RAG settings
EMBEDDING_MODEL=ai-forever/FRIDA
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_K_POINTS=3
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

# Очистка и перезапуск
docker compose -f docker/docker-compose.dev.yml down -v
docker compose -f docker/docker-compose.dev.yml up -d
```

### 🔧 Решение проблем

#### Проблемы с памятью

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

#### Проблемы с LLM API

Если возникают ошибки от OpenRouter (503, 429 и др.):

1. **Проверьте API ключ** в `.env` файле
2. **Попробуйте другую модель** - измените `default_model` в `app/rag/llm.py`
3. **Используйте чат без RAG** для тестирования
4. **Проверьте статус OpenRouter** на их сайте

#### Проблемы с Docker

```bash
# Очистка Docker
docker system prune -a

# Пересборка образов
docker compose -f docker/docker-compose.dev.yml build --no-cache

# Проверка логов
docker compose -f docker/docker-compose.dev.yml logs -f
```

## 🔄 Обновления

### Обновление зависимостей

```bash
# Пересборка образов
docker compose -f docker/docker-compose.dev.yml build --no-cache
docker compose -f docker/docker-compose.dev.yml up -d
```

### Обновление базы данных

```bash
# Применение миграций
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
