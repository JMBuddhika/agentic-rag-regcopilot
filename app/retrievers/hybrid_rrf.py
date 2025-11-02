from __future__ import annotations
from typing import List, Dict
import numpy as np
from rank_bm25 import BM25Okapi
from app.models import Embedder, CrossEncoderReranker

class HybridRetriever:
    def __init__(self, passages: List[Dict], embedder: Embedder, reranker: CrossEncoderReranker):
        # passages: [{"doc_id","section_id","url","text"}]
        self.passages = passages
        self.embedder = embedder
        self.reranker = reranker
        # BM25
        tokenized = [p["text"].lower().split() for p in passages]
        self.bm25 = BM25Okapi(tokenized)
        # Dense
        self.emb_matrix = embedder.encode([p["text"] for p in passages])
        # Precompute norms for cosine
        self.emb_norms = np.linalg.norm(self.emb_matrix, axis=1) + 1e-12

    def search(self, query: str, top_k: int = 8) -> List[Dict]:
        # ---- guard: empty corpus ----
        if not self.passages:
            return []
        # BM25 ranks
        bm_scores = self.bm25.get_scores(query.lower().split())
        bm_rank_idx = np.argsort(-bm_scores)
        # Dense ranks
        qv = self.embedder.encode([query])[0]
        qn = np.linalg.norm(qv) + 1e-12
        cos = (self.emb_matrix @ qv) / (self.emb_norms * qn)
        de_rank_idx = np.argsort(-cos)
        # Fusion (RRF) over union of top candidates
        cand = np.unique(np.concatenate([bm_rank_idx[:64], de_rank_idx[:64]]))
        bm_pos = np.argsort(np.argsort(-bm_scores[cand])) + 1
        de_pos = np.argsort(np.argsort(-cos[cand])) + 1
        rrf_scores = (1/(60+bm_pos)) + (1/(60+de_pos))
        fused_idx = cand[np.argsort(-rrf_scores)][:max(top_k*4, 32)]
        candidates = [self.passages[i] | {"bm25": float(bm_scores[i]), "dense": float(cos[i])} for i in fused_idx]
        # Rerank with cross-encoder
        scores = self.reranker.score(query, [c["text"] for c in candidates])
        order = np.argsort(-np.array(scores))[:top_k]
        return [candidates[i] | {"rerank": float(scores[i])} for i in order]
