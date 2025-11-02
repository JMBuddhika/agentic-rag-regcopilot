from __future__ import annotations
from typing import Tuple
from app.models import LLM

SYSTEM = (
"You are a Critic. Verify the DRAFT answer is fully supported by citations. "
"Rules: (1) Every factual sentence MUST have at least one inline citation like [doc_id §section]. "
"(2) If contradictions/insufficient support → mark fail. "
"Return strict JSON: {\"pass\": bool, \"issues\": [\"...\"]}."
)

def critique(llm: LLM, query: str, draft: str) -> Tuple[bool, list[str]]:
    messages = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":f"Question: {query}\n\nDRAFT:\n{draft}"}
    ]
    raw = llm.chat(messages)
    import json
    try:
        obj = json.loads(raw)
        return bool(obj.get("pass", False)), [str(x) for x in obj.get("issues", [])]
    except Exception:
        return False, ["Critic failed to parse output."]
