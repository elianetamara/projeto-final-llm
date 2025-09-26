# src/answer_agent.py
from __future__ import annotations
from typing import List, Dict
import os

import ollama
from src.prompts import get_chat_prompt, get_detector_prompt
from ollama import Client

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:7b")

def _ollama_client() -> Client:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://150.165.75.163/ollama/")
    return Client(host=base_url)


def _format_evidence(hits: List[Dict]) -> str:
    if not hits:
        return "Nenhuma evidência encontrada."
    lines = []
    for i, h in enumerate(hits, 1):
        meta = h.get("meta", {}) if "meta" in h else h
        title = meta.get("title") or meta.get("source") or f"Fonte {i}"
        url = meta.get("url") or meta.get("source_url") or ""
        page = f" p. {meta.get('page')}" if meta.get("page") else ""
        txt = (h.get("text") or "").replace("\n", " ")
        lines.append(f"[{i}] {title} ({url}{page})\n>>> {txt}")
    return "\n\n".join(lines)


def generate(user_query: str, hits: List[Dict], history: List[Dict], prompt_type: str = "chat") -> str:
    """
    Sua função original — mantida para compatibilidade.
    """
    local_context = _format_evidence(hits)
    prompt_template = get_chat_prompt() if prompt_type == "chat" else get_detector_prompt()
    prompt = prompt_template.format(query=user_query, local_context=local_context, history=history)
    
    client = _ollama_client()

    resp = client.generate(
        model=OLLAMA_MODEL,
        prompt=prompt,
        options={"num_predict": 800, "temperature": 0}
    )
    return resp["response"]

def generate_answer(user_query: str, evidences: List[Dict], history: List[Dict], prompt_type: str = "chat") -> str:
    """
    Nome que o agent_graph.py importa. Apenas delega para `generate`.
    """
    return generate(user_query=user_query, hits=evidences, history=history, prompt_type=prompt_type)
