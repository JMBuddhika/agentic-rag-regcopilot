from __future__ import annotations
from typing import Dict, List
from dataclasses import dataclass

from app.typing import AgentState, Evidence
from app.models import LLM, Embedder, CrossEncoderReranker
from app.retrievers.hybrid_rrf import HybridRetriever
from app.agents.planner import plan
from app.agents.retriever import EvidenceCollector
from app.agents.answerer import answer_subq, merge_answers
from app.agents.critic import critique
from app.agents.repair import propose_new_queries
from app.config import SETTINGS

@dataclass
class Runtime:
    llm: LLM
    retriever: HybridRetriever

def load_corpus(db_path: str) -> List[Dict]:
    import json
    from pathlib import Path
    corpus_p = Path(db_path)/"corpus.jsonl"
    passages = [json.loads(l) for l in corpus_p.read_text(encoding="utf-8").splitlines()] if corpus_p.exists() else []
    return passages

def build_runtime() -> Runtime:
    llm = LLM()
    emb = Embedder()
    rer = CrossEncoderReranker()
    passages = load_corpus(SETTINGS.db_path)
    retr = HybridRetriever(passages=passages, embedder=emb, reranker=rer)
    return Runtime(llm=llm, retriever=retr)

class AgenticRAG:
    def __init__(self, rt: Runtime):
        self.rt = rt
        self.collector = EvidenceCollector(rt.retriever)

    def run(self, query: str, top_k: int = 6, max_loops: int = 2) -> Dict:
        state: AgentState = {"query": query, "evidences": {}}
        # 1) Plan
        subs = plan(self.rt.llm, query)
        state["sub_questions"] = subs
        # 2) Retrieve for each subq
        sub_answers: Dict[str, str] = {}
        for s in subs:
            evs = self.collector.collect(s, top_k=top_k)
            state["evidences"][s] = evs
            sub_answers[s] = answer_subq(self.rt.llm, query, s, evs)
        # 3) Merge
        draft = merge_answers(self.rt.llm, query, sub_answers)
        state["draft_answer"] = draft
        # 4) Critique & repair loop
        for _ in range(max_loops):
            ok, issues = critique(self.rt.llm, query, draft)
            if ok:
                state["final_answer"] = draft
                state["needs_repair"] = False
                return state
            # propose new queries and collect new evidence
            new_qs = propose_new_queries(self.rt.llm, query, issues)
            for q in new_qs:
                evs = self.collector.collect(q, top_k=top_k)
                state["evidences"].setdefault(q, []).extend(evs)
            # Re-merge with augmented evidence
            for q in new_qs:
                sub_answers[q] = answer_subq(self.rt.llm, query, q, state["evidences"][q])
            draft = merge_answers(self.rt.llm, query, sub_answers)
        state["final_answer"] = draft
        state["needs_repair"] = True
        return state
