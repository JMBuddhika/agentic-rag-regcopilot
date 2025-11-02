
# Regulatory & Policy Copilot — Agentic RAG (Multi-hop, Section-Cited)

An agentic RAG that answers complex compliance questions from regulatory PDFs/HTML with a **planner → retriever → answerer → critic → repair** loop, **hybrid retrieval (BM25 + dense + cross-encoder rerank)**, and **inline section-level citations** like `[privacy_act_2024 §3.1]`.


## 🚀 Quickstart (uv + Groq)

```bash
# 0) Clone and enter the repo
# git clone https://github.com/yourname/agentic-rag-regcopilot && cd agentic-rag-regcopilot

# 1) Create & activate env
uv venv .venv && source .venv/bin/activate   # Windows: .\.venv\Scripts\activate

# 2) Install deps
uv sync

# 3) Configure API key
cp .env.example .env           # Windows: copy .env.example .env
# edit .env and set: GROQ_API_KEY=your_key_here

# 4) (Optional) Add sample docs
# Unzip regulatory_sample_docs.zip into data/sample/ or use your own PDFs/HTML

# 5) Ingest docs (build vectorstore + embeddings)
uv run -m app.ingest.ingest --input_dir data/sample --db_path .vectorstore

# 6) Run API
uv run uvicorn app.api:app --reload --port 8000
````

**Try a query**

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Is cross-border transfer allowed without explicit consent?", "top_k": 8}'
```

**Optional UI**

```bash
uv run streamlit run app/ui.py
```

---

## 🧠 What makes it “agentic”

* **Planner** decomposes the user question into verifiable sub-questions.
* **Hybrid Retrieval**: BM25 + sentence-transformer embeddings → RRF fusion → **bge-reranker** cross-encoder.
* **Answerer** writes ONLY from retrieved evidence with **inline citations**.
* **Critic/Judge** verifies every factual sentence has a citation; on failure, **Repair** suggests new sub-queries and retries.
* **Abstain** behavior for insufficient evidence.

---

## 🧱 Project Structure

```
app/
  agents/
    planner.py        # make sub-questions
    retriever.py      # collect top-k evidence per subq
    answerer.py       # grounded sub-answers + merge
    critic.py         # check citations/consistency
    repair.py         # propose new queries if fail
  ingest/
    loaders.py        # PDF/HTML loaders + chunking
    ingest.py         # build corpus.jsonl + embeddings
  retrievers/
    hybrid_rrf.py     # BM25 + dense + cross-encoder rerank
  api.py              # FastAPI endpoint /ask
  config.py           # env + model names
  graph.py            # runtime wiring + agentic loop
  models.py           # Groq LLM, embeddings, reranker
  typing.py           # shared types
  ui.py               # tiny Streamlit UI (optional)
data/
  sample/             # put your PDFs/HTML here
.vectorstore/         # generated on ingest (corpus + embeddings)
pyproject.toml        # uv project config
.env.example          # set GROQ_API_KEY
```

---

## ⚙️ Configuration

**Environment (.env)**

```
GROQ_API_KEY=your_groq_key
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=BAAI/bge-reranker-base
DB_PATH=.vectorstore
```

**Models**

* LLM: `llama-3.3-70b-versatile` via **Groq**
* Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (swap to `bge-m3` for higher recall)
* Reranker: `BAAI/bge-reranker-base` (cross-encoder)

---

## 📥 Add Documents

Drop your **regulatory PDFs/HTML** into `data/sample/` (e.g., acts, guidelines, regulator FAQs).

Sample synthetic docs included:

* `privacy_act_2024.html`
* `cross_border_transfer_code_2023.html`
* `regulator_faq_consent.html`
* `payments_kyc_rules_2022.html`
* `dpia_guideline_2023.html`

Re-ingest whenever you add/update content:

```bash
uv run -m app.ingest.ingest --input_dir data/sample --db_path .vectorstore
```

---

## 🧪 Example Questions

* “Is cross-border transfer allowed without explicit consent?”
  → cites `§3.1` / safeguards & consent requirements.

* “How long must we keep KYC records after off-boarding?”
  → cites KYC retention section.

* “Do we need a DPIA for large-scale cross-border analytics?”
  → cites DPIA triggers + minimum contents.

---

## 📊 Evaluation (hooks)

A placeholder `evals/ragas_suite.py` is included. Suggested metrics:

* **RAGAS**: faithfulness, answer relevancy, context precision/recall
* **Latency & Cost**: p50/p95 + token usage
* **Drift checks**: re-run evals after each crawl/re-ingest

Add a small gold set under `data/evals/` to compare:

* **Baseline RAG** vs **Agentic (with Critic/Repair)**

---

## 🛠️ Troubleshooting

* **`ZeroDivisionError` in BM25** → ingest ran on an empty folder. Ensure `data/sample/` has files, then re-ingest.
* **`.env not loaded`** → we `load_dotenv()` in `app/config.py`. Verify `.env` exists and the key name is `GROQ_API_KEY`.
* **Large download for reranker (1.1GB)** → first run is slow; it caches under `~/.cache/huggingface/`.
* **Windows symlink warning from HF Hub** → optional; enable Developer Mode or ignore the warning.
* **Slow answers** → reduce `top_k`, switch to a lighter reranker, or use smaller embedding model.

---

## 🧩 Roadmap Ideas

* Freshness guard (min source date per answer)
* Exact span highlighting in source viewer
* Multi-tenant doc spaces + access control
* Periodic crawler + re-embed deltas
* Automated evals in CI (TruLens/RAGAS + thresholds)

---

## 📜 License

MIT 

---

