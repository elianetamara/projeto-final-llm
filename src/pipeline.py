# src/pipeline.py
from __future__ import annotations
import os
from typing import Literal, Optional, List, Dict, Any

from langchain_community.tools.tavily_search import TavilySearchResults

from src.validator import check_coverage, add_disclaimer
from src.answer_agent import generate
from src.retriever import PDFIndexerRetriever


# Singleton simples para não reinstanciar o retriever a cada pergunta
_retriever: Optional[PDFIndexerRetriever] = None
def get_retriever() -> PDFIndexerRetriever:
    global _retriever
    if _retriever is None:
        _retriever = PDFIndexerRetriever()
        # Se quiser indexar automaticamente em dev:
        # _retriever.build_index_from_folder("data/pdfs")
    return _retriever


def _maybe_web_search(query: str, k: int = 5):
    import os, logging
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        raw = TavilySearchResults(k=k).run(query)
        if isinstance(raw, str):
            raw = [raw]

        items = []
        for r in raw:
            if isinstance(r, str):
                txt = r
                src = "web"
            elif isinstance(r, dict):
                # tente pegar conteúdo/trecho; ajuste conforme o schema retornado
                txt = r.get("content") or r.get("snippet") or r.get("text") or r.get("title") or ""
                src = r.get("url") or r.get("source") or "web"
            else:
                continue
            if not txt:
                continue
            items.append({"text": str(txt), "meta": {"source": str(src)}})
        return items
    except Exception as e:
        logging.exception("Tavily search failed: %s", e)
        return []


def _merge_hits(local_hits, web_hits):
    seen = set()
    merged = []
    for h in (local_hits + web_hits):
        key = str(h.get("text", ""))[:5000]  # string sempre
        if key not in seen:
            seen.add(key)
            merged.append(h)
    return merged



def run_pipeline(
    mode: Literal["chat", "detector"],
    user_input: str,
    history: Optional[List[Dict[str, str]]] = None
) -> str:
    retr = get_retriever()

    # 1) Busca local (Chroma)
    hits = retr.retrieve(user_input, k=5)

    # 2) Opcional: busca web apenas no modo detector
    web_hits = _maybe_web_search(user_input, k=5) if mode == "detector" else []

    # 3) Constrói evidências para o prompt
    prompt_hits = _merge_hits(hits, web_hits)

    # 4) Gera resposta (alinha prompt_type com o modo)
    raw_answer = generate(
        user_input,
        prompt_hits,
        history or [],
        prompt_type=mode  # "chat" ou "detector"
    )

    # 5) Verifica cobertura/citações e adiciona disclaimer
    # problems = check_coverage(raw_answer)
    # if problems:
    raw_answer += "\n\n⚠️ Algumas sentenças podem estar sem referência adequada."
    final_answer = add_disclaimer(raw_answer)

    return final_answer
