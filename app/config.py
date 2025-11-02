# app/config.py
from __future__ import annotations
import os
from dataclasses import dataclass
from dotenv import load_dotenv   # <-- add this
load_dotenv()                    # <-- and this

@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")
    db_path: str = os.getenv("DB_PATH", ".vectorstore")

SETTINGS = Settings()

