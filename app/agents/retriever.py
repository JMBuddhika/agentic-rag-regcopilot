from __future__ import annotations
from typing import List, Dict
from app.retrievers.hybrid_rrf import HybridRetriever
from app.typing import Evidence

class EvidenceCollector:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
    def collect(self, subq: str, top_k: int = 6) -> List[Evidence]:
        hits = self.retriever.search(subq, top_k=top_k)
        ev: List[Evidence] = []
        for h in hits:
            ev.append({
                "doc_id": h.get("doc_id","unknown"),
                "section_id": h.get("section_id","?"),
                "url": h.get("url"),
                "text": h.get("text",""),
            })
        return ev
