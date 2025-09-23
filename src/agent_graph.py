from __future__ import annotations
from typing import TypedDict, Literal, List, Dict, Any
import os

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.retriever import PDFIndexerRetriever
from src.answer_agent import generate_answer
from src.selfcheck import extract_claims, check_claims_and_rewrite


class State(TypedDict, total=False):
    question: str
    mode: Literal["chat", "detector"]
    history: List[Dict[str, str]]
    intent: Literal["qa", "detector"]
    evidences: List[Dict[str, Any]]
    draft: str
    final: str


SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.50"))


def _sim_of(hit: Dict[str, Any]) -> float:
    """Similaridade do hit. Se não houver info, trate como 0.0 (não confie)."""
    if not hit:
        return 0.0
    sim = hit.get("similarity")
    if sim is not None:
        try:
            return float(sim)
        except Exception:
            return 0.0
    dist = hit.get("distance")
    if dist is not None:
        try:
            return 1.0 - float(dist)  
        except Exception:
            return 0.0
    return 0.0


def _router(state: State) -> Dict[str, Any]:
    q = (state.get("question") or "").lower()
    mode = state.get("mode") or "chat"
    if mode == "detector" or any(t in q for t in ["é golpe", "fraude", "suspeita", "phishing", "link", "sms"]):
        return {"intent": "detector"}
    return {"intent": "qa"}


def _retrieve(state: State, retriever: PDFIndexerRetriever) -> Dict[str, Any]:
    q = state["question"]

    local_hits = retriever.retrieve(q, k=12)
    filtered = [h for h in local_hits if _sim_of(h) >= SIM_THRESHOLD]

    evidences: List[Dict[str, Any]] = []
    seen = set()
    for h in filtered:
        meta = h.get("meta") or {}
        key = (meta.get("source"), meta.get("page"), h.get("id"))
        if key in seen:
            continue
        seen.add(key)
        evidences.append({
            "id": h.get("id"),
            "text": " ".join((h.get("text") or "").split()),
            "title": meta.get("title") or meta.get("source"),
            "url": meta.get("url"),
            "page": meta.get("page"),
            "score": _sim_of(h),
            "origin": "Local",
        })
        if len(evidences) >= 8:
            break

    return {"evidences": evidences}


def _answer(state: State) -> Dict[str, Any]:
    evidences = state.get("evidences") or []
    if not evidences:
        return {"draft": "NÃO ENCONTREI BASE"} 
    draft = generate_answer(
        user_query=state["question"],
        evidences=evidences,
        history=state.get("history") or [],
        prompt_type=("detector" if state.get("intent") == "detector" else "chat"),
    )
    return {"draft": draft}


def _selfcheck(state: State, retriever: PDFIndexerRetriever) -> Dict[str, Any]:
    draft = state.get("draft") or ""
    if not draft.strip():
        return {"final": "NÃO ENCONTREI BASE"}

    claims = extract_claims(draft)
    final = check_claims_and_rewrite(
        draft=draft,
        claims=claims,
        retriever=retriever,
        min_sim=SIM_THRESHOLD,       
        min_overlap_terms=1,         
    )
    return {"final": final}


def _safety(state: State) -> Dict[str, Any]:
    return {}


def build_app(retriever: PDFIndexerRetriever):
    g = StateGraph(State)
    g.add_node("router", _router)
    g.add_node("retrieve", lambda s: _retrieve(s, retriever))
    g.add_node("answer", _answer)
    g.add_node("selfcheck", lambda s: _selfcheck(s, retriever))
    g.add_node("safety", _safety)

    g.set_entry_point("router")
    g.add_edge("router", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "selfcheck")
    g.add_edge("selfcheck", "safety")
    g.add_edge("safety", END)

    return g.compile(checkpointer=MemorySaver())
