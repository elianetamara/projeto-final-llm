from __future__ import annotations
from typing import List, Dict, Any
import re

from src.retriever import PDFIndexerRetriever

def _sentences(text: str) -> List[str]:
    parts = re.split(r'(?<=[\.\!\?])\s+', (text or "").strip())
    return [p.strip() for p in parts if p.strip()]

def _overlap(a: str, b: str) -> int:
    toks_a = set(w.lower() for w in re.findall(r"\w+", a or ""))
    toks_b = set(w.lower() for w in re.findall(r"\w+", b or ""))
    return len(toks_a & toks_b)

def _sim_of(hit: Dict[str, Any]) -> float:
    if not hit:
        return 0.0
    sim = hit.get("similarity")
    if sim is not None:
        try:
            return float(sim)
        except Exception:
            return 0.0
    dist = hit.get("distance")
    return (1.0 - float(dist)) if dist is not None else 0.0

def extract_claims(text: str) -> List[str]:
    sents = _sentences(text)
    return [s for s in sents if len(re.findall(r"\w+", s)) >= 5][:20]

def _find_supports_for_claim(
    claim: str,
    retriever: PDFIndexerRetriever,
    min_sim: float = 0.50,
    min_overlap_terms: int = 1,
    k: int = 6,
) -> List[Dict[str, Any]]:
    hits = retriever.retrieve(claim, k=k)
    if not hits:
        return []
    accepted = [h for h in hits if _sim_of(h) >= min_sim and _overlap(claim, h.get("text", "")) >= min_overlap_terms]
    accepted.sort(key=_sim_of, reverse=True)
    return accepted[:2]

def check_claims_and_rewrite(
    draft: str,
    claims: List[str],
    retriever: PDFIndexerRetriever,
    min_sim: float = 0.50,
    min_overlap_terms: int = 1,
) -> str:
    kept_lines: List[str] = []
    for sent in _sentences(draft):
        supports = _find_supports_for_claim(
            claim=sent,
            retriever=retriever,
            min_sim=min_sim,
            min_overlap_terms=min_overlap_terms,
            k=6,
        )
        if not supports:
            continue  

        ev = supports[0]
        meta = ev.get("meta", {})
        title = meta.get("title") or meta.get("source") or "Fonte"
        url = meta.get("url")
        page = meta.get("page")

        if url:
            cite = f" [{title} ({url})]"
        elif page:
            cite = f" [{title}, p. {page}]"
        else:
            cite = f" [{title}]"

        line = sent if cite in sent else (sent + cite)
        kept_lines.append(line)

    final = "\n".join(kept_lines).strip()
    return final if final else "NÃO ENCONTREI BASE"
