from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")
    PROJECT_NAME: str = "Wryte Backend"
    VERSION: str = "0.1.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"

    DATABASE_URL: str
    NEXTAUTH_SECRET: str 
    NVIDIA_API_KEY: str 
    AUTH_URL: str
    TINYFISH_API_KEY: str

    CORE_ORIGINS: list[str]=[
        "http://localhost:3000",
        "https://wryte-ti.vercel.app",
    ]


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
