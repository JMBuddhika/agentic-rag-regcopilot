from __future__ import annotations
from typing import List
from app.config import SETTINGS

# --- LLM client (Groq only) ---
class LLM:
    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        if not SETTINGS.groq_api_key:
            raise RuntimeError("Set GROQ_API_KEY in your .env")
        from groq import Groq
        self.client = Groq(api_key=SETTINGS.groq_api_key)
        self.model = model

    def chat(self, messages: List[dict], temperature: float = 0.2) -> str:
        chat = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=messages,
        )
        return chat.choices[0].message.content

# --- Embeddings ---
from sentence_transformers import SentenceTransformer
import numpy as np

class Embedder:
    def __init__(self, name: str = SETTINGS.embed_model):
        self.model = SentenceTransformer(name)
    def encode(self, texts: List[str]) -> np.ndarray:
        return np.array(self.model.encode(texts, batch_size=64, show_progress_bar=False, convert_to_numpy=True))

# --- Cross-encoder reranker ---
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class CrossEncoderReranker:
    def __init__(self, name: str = SETTINGS.rerank_model):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name)
        self.model.eval()
    @torch.inference_mode()
    def score(self, query: str, passages: List[str]) -> List[float]:
        inputs = self.tokenizer([query]*len(passages), passages, padding=True, truncation=True, return_tensors="pt")
        logits = self.model(**inputs).logits.squeeze(-1)
        return logits.detach().cpu().tolist()
