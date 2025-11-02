from __future__ import annotations
import argparse
from pathlib import Path
import json
from app.config import SETTINGS
from app.models import Embedder
from app.ingest.loaders import load_pdf, load_html

# We persist a simple JSONL corpus + a cached embeddings matrix (npy)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--db_path", type=str, default=SETTINGS.db_path)
    p.add_argument("--chunk_size", type=int, default=900)
    p.add_argument("--chunk_overlap", type=int, default=120)
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    db = Path(args.db_path)
    db.mkdir(parents=True, exist_ok=True)

    passages = []
    for path in input_dir.rglob("*"):
        if path.suffix.lower() in {".pdf"}:
            passages += load_pdf(path)
        elif path.suffix.lower() in {".html",".htm"}:
            passages += load_html(path)

    # Save corpus
    (db / "corpus.jsonl").write_text("\n".join(json.dumps(p) for p in passages), encoding="utf-8")

    # Pre-embed for faster startup (optional)
    emb = Embedder()
    import numpy as np
    mat = emb.encode([p["text"] for p in passages])
    np.save(db / "embeddings.npy", mat)

    print(f"Ingested {len(passages)} passages → {db}")

if __name__ == "__main__":
    main()
