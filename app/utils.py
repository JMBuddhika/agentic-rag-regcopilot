from __future__ import annotations
import re
from typing import List

def rrf(ranks: List[List[int]], k: int = 60) -> List[float]:
    # Reciprocal Rank Fusion over multiple rank lists of same length
    # Return score per candidate index (assumes same candidate order)
    import numpy as np
    ranks_np = np.array(ranks)
    return (1.0 / (k + ranks_np)).sum(axis=0).tolist()

SECTION_ID_RE = re.compile(r"§?\s*([\w\.\-]+)\s*(\([a-zA-Z0-9]+\))?")

def normalize_section_id(text: str) -> str:
    m = SECTION_ID_RE.search(text)
    return (m.group(0).strip() if m else "§?")
