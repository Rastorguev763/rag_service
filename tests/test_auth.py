import pytest
from fastapi import status


@pytest.mark.asyncio
@pytest.mark.auth
class TestAuthEndpoints:
    """Тесты для endpoints аутентификации"""

    async def test_register_user_success(self, client):
        """Тест успешной регистрации пользователя"""
        user_data = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "newpassword123",
        }

        response = client.post("/auth/register", json=user_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == user_data["username"]
        assert data["email"] == user_data["email"]
        assert "id" in data
        assert "hashed_password" not in data

    async def test_register_user_duplicate_username(self, client, test_user):
        """Тест регистрации с существующим username"""
        user_data = {
            "username": test_user.username,
            "email": "different@example.com",
            "password": "password123",
        }

        response = client.post("/auth/register", json=user_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Username already registered" in response.json()["detail"]

    async def test_register_user_duplicate_email(self, client, test_user):
        """Тест регистрации с существующим email"""
        user_data = {
            "username": "differentuser",
            "email": test_user.email,
            "password": "password123",
        }

        response = client.post("/auth/register", json=user_data)

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        assert "Email already registered" in response.json()["detail"]

    async def test_register_user_invalid_data(self, client):
        """Тест регистрации с некорректными данными"""
        user_data = {"username": "user", "email": "invalid-email"}

        response = client.post("/auth/register", json=user_data)

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    async def test_login_success(self, client, test_user):
        """Тест успешной аутентификации"""
        form_data = {"username": test_user.username, "password": "testpassword"}

        response = client.post("/auth/token", data=form_data)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    async def test_login_invalid_username(self, client):
        """Тест аутентификации с несуществующим пользователем"""
        form_data = {"username": "nonexistent", "password": "password123"}

        response = client.post("/auth/token", data=form_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect username or password" in response.json()["detail"]

    async def test_login_invalid_password(self, client, test_user):
        """Тест аутентификации с неправильным паролем"""
        form_data = {"username": test_user.username, "password": "wrongpassword"}

        response = client.post("/auth/token", data=form_data)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Incorrect username or password" in response.json()["detail"]

    async def test_get_current_user_success(self, client, auth_headers, test_user):
        """Тест получения информации о текущем пользователе"""
        response = client.get("/auth/me", headers=auth_headers)

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["username"] == test_user.username
        assert data["email"] == test_user.email
        assert data["id"] == test_user.id

    async def test_get_current_user_no_token(self, client):
        """Тест получения информации о пользователе без токена"""
        response = client.get("/auth/me")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_current_user_invalid_token(self, client):
        """Тест получения информации о пользователе с недействительным токеном"""
        headers = {"Authorization": "Bearer invalid_token"}
        response = client.get("/auth/me", headers=headers)

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    async def test_get_current_user_inactive_user(self, client, db_session, test_user):
        """Тест получения информации о неактивном пользователе"""
        # Делаем пользователя неактивным
        test_user.is_active = False
        db_session.add(test_user)
        await db_session.commit()

        # Пытаемся получить токен для неактивного пользователя
        response = client.post(
            "/auth/token",
            data={"username": test_user.username, "password": "testpassword"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
