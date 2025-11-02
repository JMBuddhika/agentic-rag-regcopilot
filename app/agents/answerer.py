from __future__ import annotations
from typing import List, Dict
from app.models import LLM
from app.typing import Evidence

SYSTEM = (
"You are a Regulatory Answerer. Answer ONLY with information supported by the provided EVIDENCE. "
"Cite inline using [doc_id §section_id]. If evidence is insufficient, say 'Insufficient evidence'. "
)

TEMPLATE = (
"User question: {query}\n\n"
"Sub-question: {subq}\n\n"
"EVIDENCE (each item = text + [doc_id §section_id]):\n{evidence}\n\n"
"Write a concise answer (3-6 sentences) grounded ONLY in the evidence."
)

def render_evidence(evs: List[Evidence]) -> str:
    out = []
    for e in evs:
        cite = f"[{e['doc_id']} §{e['section_id']}]" if e.get('section_id') else f"[{e['doc_id']}]"
        out.append(f"- {e['text']}\n  {cite}")
    return "\n".join(out)

def answer_subq(llm: LLM, query: str, subq: str, evs: List[Evidence]) -> str:
    content = TEMPLATE.format(query=query, subq=subq, evidence=render_evidence(evs))
    messages = [
        {"role":"system","content":SYSTEM},
        {"role":"user","content":content}
    ]
    return llm.chat(messages)

def merge_answers(llm: LLM, query: str, sub_answers: Dict[str,str]) -> str:
    messages = [
        {"role":"system","content":(
            "You merge grounded sub-answers into a single cohesive response. "
            "Keep inline citations from sub-answers. Add a short 'Risk/Notes' bullet if applicable."
        )},
        {"role":"user","content":(
            f"Question: {query}\n\nSub-answers (JSON):\n{sub_answers}\n\n"
            "Combine them into a final answer with inline citations."
        )}
    ]
    return llm.chat(messages)
