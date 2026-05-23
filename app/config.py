from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    PROJECT_NAME: str = "Wryte Backend"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str
    NVIDIA_API_KEY: str 
    TINYFISH_API_KEY: str

    CORS_ORIGINS: list[str]=[
        "http://localhost:3000",
        "https://wryte-ti.vercel.app",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
