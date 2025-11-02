# save as app/ui.py and run: uv run streamlit run app/ui.py
import streamlit as st
import requests

st.set_page_config(page_title="Regulatory Copilot", layout="wide")
st.title("Regulatory & Policy Copilot — Agentic RAG")

q = st.text_input("Ask a question", "Is cross-border transfer allowed without explicit consent?")
if st.button("Ask"):
    with st.spinner("Reasoning..."):
        r = requests.post("http://localhost:8000/ask", json={"query": q, "top_k": 8}).json()
    st.subheader("Answer")
    st.write(r["final_answer"]) 
    st.subheader("Sub-questions")
    st.write(r["sub_questions"]) 
    st.subheader("Evidence")
    for k, evs in r["evidences"].items():
        with st.expander(k):
            for e in evs:
                st.markdown(f"**{e['doc_id']} §{e['section_id']}** — {e['text'][:500]}…")
