from __future__ import annotations
from typing import TypedDict, List, Dict, Any

class Evidence(TypedDict):
    doc_id: str
    section_id: str
    url: str | None
    text: str

class AgentState(TypedDict, total=False):
    query: str
    sub_questions: List[str]
    evidences: Dict[str, List[Evidence]]  # by sub_question
    draft_answer: str
    final_answer: str
    needs_repair: bool
    messages: List[Dict[str, Any]]  # for trace
