from __future__ import annotations
from typing import List
from app.models import LLM

SYSTEM = (
    "You are a planning agent for regulatory Q&A. "
    "Break the user query into 2-5 sub-questions that can be verified against statutes/sections. "
    "Return only a JSON list of sub-questions, concise and non-overlapping."
)

def plan(llm: LLM, query: str) -> List[str]:
    msg = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":f"Query: {query}\nReturn JSON list only."}
    ]
    raw = llm.chat(msg)
    import json
    try:
        subs = json.loads(raw)
        assert isinstance(subs, list) and all(isinstance(x, str) for x in subs)
        return subs[:5]
    except Exception:
        # Fallback: naive split on punctuation
        return [query]
