#!/bin/bash
set -e

echo "Waiting for database to be ready..."
# Ждем, пока база данных будет готова
until alembic current > /dev/null 2>&1; do
    echo "Database not ready, waiting..."
    sleep 2
done

echo "Applying database migrations..."
# Применяем миграции
alembic upgrade head

echo "Starting application..."
# Запускаем приложение
exec "$@" 