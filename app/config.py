from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Genesis AI"
    debug: bool = False
    database_url: str = ""
    redis_url: str = ""
    llm_api_key: str = ""
    secret_key: str = "change-me"
    class Config:
        env_file = ".env"

settings = Settings()
