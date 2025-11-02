from __future__ import annotations
from typing import List
from app.models import LLM

SYSTEM = (
"You are a Repair agent. For each listed issue, propose improved sub-queries to fetch better evidence. "
"Return a JSON list of 1-3 queries."
)

def propose_new_queries(llm: LLM, query: str, issues: List[str]) -> List[str]:
    messages = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":f"User question: {query}\nIssues: {issues}\nReturn JSON list only."}
    ]
    import json
    raw = llm.chat(messages)
    try:
        arr = json.loads(raw)
        if isinstance(arr, list):
            return [str(x) for x in arr][:3]
    except Exception:
        pass
    return []
