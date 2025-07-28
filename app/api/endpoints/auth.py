# TODO
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.utils.auth import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
)
from app.config.config import settings
from app.db.database import get_db
from app.utils.logger import logger
from app.models.models import User
from app.schemas.schemas import Token
from app.schemas.schemas import User as UserSchema
from app.schemas.schemas import UserCreate

router = APIRouter()


@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)
):
    """Получение JWT токена для аутентификации"""
    try:
        user = authenticate_user(db, form_data.username, form_data.password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )

        logger.info(f"Пользователь {user.username} успешно аутентифицирован")
        return {"access_token": access_token, "token_type": "bearer"}

    except Exception as e:
        logger.error(f"Ошибка аутентификации: {e}")
        raise


@router.post("/register", response_model=UserSchema)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Регистрация нового пользователя"""
    try:
        from app.utils.auth import get_password_hash

        # Проверяем, что пользователь не существует
        existing_user = (
            db.query(User).filter(User.username == user_data.username).first()
        )
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered",
            )

        existing_email = db.query(User).filter(User.email == user_data.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Создаем нового пользователя
        hashed_password = get_password_hash(user_data.password)
        user = User(
            username=user_data.username,
            email=user_data.email,
            hashed_password=hashed_password,
        )

        db.add(user)
        db.commit()
        db.refresh(user)

        logger.info(f"Зарегистрирован новый пользователь: {user.username}")
        return user

    except Exception as e:
        logger.error(f"Ошибка регистрации: {e}")
        db.rollback()
        raise


@router.get("/me", response_model=UserSchema)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Получение информации о текущем пользователе"""
    return current_user
