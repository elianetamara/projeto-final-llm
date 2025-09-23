# src/safety.py
from __future__ import annotations

def add_disclaimer(text: str, mode: str = "chat") -> str:
    base = "\n\n> **Aviso**: Conteúdo educativo. Não substitui canais oficiais do seu banco/autoridades. Em caso de suspeita, acione seu banco pelo app e registre ocorrência conforme orientação local."
    if "NÃO ENCONTREI BASE" in text:
        return text + base
    return text + base
