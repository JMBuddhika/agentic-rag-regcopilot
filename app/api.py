from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from app.graph import build_runtime, AgenticRAG

app = FastAPI(title="Regulatory & Policy Copilot — Agentic RAG")
RT = build_runtime()
ENGINE = AgenticRAG(RT)

class AskIn(BaseModel):
    query: str
    top_k: int = 8

@app.post("/ask")
def ask(body: AskIn):
    result = ENGINE.run(body.query, top_k=body.top_k)
    return {
        "query": result.get("query"),
        "sub_questions": result.get("sub_questions"),
        "final_answer": result.get("final_answer"),
        "needs_repair": result.get("needs_repair", False),
        "evidences": result.get("evidences"),
    }
