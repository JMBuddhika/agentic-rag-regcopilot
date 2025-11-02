from __future__ import annotations
from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from bs4 import BeautifulSoup

# Simple sectionizer: split paragraphs into ~900-char chunks and synthesize section ids

def load_pdf(path: Path) -> List[Dict]:
    reader = PdfReader(str(path))
    chunks: List[Dict] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for j, part in enumerate(_chunk(text, 900, 120)):
            chunks.append({
                "doc_id": path.stem,
                "section_id": f"p{i}.{j}",
                "url": None,
                "text": part.strip(),
            })
    return chunks

def load_html(path: Path) -> List[Dict]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    texts = [p.get_text(" ", strip=True) for p in soup.find_all(["p","li","div","section"]) if p.get_text(strip=True)]
    chunks: List[Dict] = []
    for idx, t in enumerate(texts):
        for j, part in enumerate(_chunk(t, 900, 120)):
            chunks.append({
                "doc_id": path.stem,
                "section_id": f"s{idx}.{j}",
                "url": None,
                "text": part.strip(),
            })
    return chunks

def _chunk(text: str, size: int, overlap: int) -> List[str]:
    text = " ".join(text.split())
    out, start = [], 0
    while start < len(text):
        out.append(text[start:start+size])
        start += size - overlap
    return out
