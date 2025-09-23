#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera gabarito rápido via sua pipeline para perguntas em JSONL/JSON.
Não inicializa retriever aqui (assume que sua pipeline já está pronta).
"""

from __future__ import annotations
import os
import re
import json
import argparse
from typing import List, Dict, Any

# --- bootstrap de path para permitir "from src.*" ao rodar de qualquer lugar ---
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]  # diretório-pai de src/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------------------------------------

# Importa APENAS a pipeline (que já deve saber usar o retriever pronto)
from src.pipeline import run_pipeline


# ---------- Utilidades de IO ----------
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Aceita JSONL (1 obj/linha) ou um array JSON. Ignora linhas vazias/comentários."""
    import io
    with io.open(path, "r", encoding="utf-8-sig") as f:
        text = f.read().strip()
        if not text:
            return []
        if text[0] == "[":  # array JSON
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("Arquivo JSON deve conter uma lista.")
            return data
        # JSONL
        items = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            items.append(json.loads(line))
        return items


def dump_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ---------- Resumo curto (gabarito rápido) ----------
def cleanup(text: str) -> str:
    t = re.sub(r"⚠️.*$", "", text, flags=re.MULTILINE).strip()
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"(?:Fontes?|Referências?):.*$", "", t, flags=re.IGNORECASE).strip()
    return t


def summarize_to_hint(text: str, max_chars: int = 220) -> str:
    """
    Heurística simples:
    - Pega bullets (3 primeiras) OU 1–2 períodos iniciais;
    - Encurta para máx. `max_chars`.
    """
    t = cleanup(text)

    bullets = re.findall(r"(?:^|\n)[\-\•\–]\s*(.+)", t)
    if bullets:
        hint = "; ".join(b.strip() for b in bullets[:3])
        return (hint[:max_chars] + "…") if len(hint) > max_chars else hint

    sentences = re.split(r"(?<=[\.\!\?])\s+", t)
    if not sentences:
        return t[:max_chars]

    take = 2 if len(sentences) > 1 else 1
    hint = " ".join(sentences[:take]).strip()
    if len(hint) > max_chars:
        hint = hint[:max_chars].rstrip() + "…"
    return hint


# ---------- Execução em lote ----------
def intent_to_mode(intent: str) -> str:
    return "detector" if (intent or "").strip().lower() == "detector" else "chat"


def main():
    ap = argparse.ArgumentParser(description="Gera gabarito rápido via pipeline a partir de perguntas JSONL/JSON.")
    ap.add_argument("--in", dest="inp", required=True, help="Arquivo de perguntas (JSONL ou array JSON).")
    ap.add_argument("--out", dest="outp", required=True, help="Arquivo de saída JSONL com respostas.")
    ap.add_argument("--force-mode", choices=["chat", "detector"], default=None,
                    help="Força o modo da pipeline; se ausente, usa o intent de cada pergunta.")
    args = ap.parse_args()

    rows = load_jsonl(args.inp)
    if not rows:
        print("Nenhuma pergunta encontrada.")
        return

    out_rows: List[Dict[str, Any]] = []
    for obj in rows:
        q = obj.get("question", "")
        intent = obj.get("intent", "chat")
        mode = args.force_mode or intent_to_mode(intent)

        try:
            full_answer = run_pipeline(mode=mode, user_input=q, history=[])
        except Exception as e:
            full_answer = f"[ERRO NA PIPELINE] {e}"

        hint = summarize_to_hint(full_answer)

        new_obj = dict(obj)
        new_obj["pipeline_answer"] = full_answer
        new_obj["answer_hint"] = hint
        out_rows.append(new_obj)

    dump_jsonl(args.outp, out_rows)
    print(f"OK! Gravado em: {args.outp}")


if __name__ == "__main__":
    main()
