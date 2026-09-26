"""
App configuration read from environment variables (see .env.example).

Uses a plain stdlib dataclass instead of pydantic-settings — one less
dependency, and simple enough here that pydantic-settings wouldn't add
much beyond what os.getenv + defaults already give us.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env loading is a convenience; env vars set another way still work


@dataclass(frozen=True)
class Settings:
    database_path: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_PATH",
            str(Path(__file__).resolve().parents[3] / "documind.db"),
        )
    )
    max_upload_size: int = field(
        default_factory=lambda: int(os.getenv("MAX_UPLOAD_SIZE", 5 * 1024 * 1024))
    )
    model_dir: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_DIR",
            str(Path(__file__).resolve().parents[3] / "ml" / "model"),
        )
    )
    cors_origins: list = field(
        default_factory=lambda: os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    )


settings = Settings()
