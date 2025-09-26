from __future__ import annotations
from typing import TypedDict, Literal, List, Dict, Any, Optional
import os, re, unicodedata

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.retriever import PDFIndexerRetriever
from src.answer_agent import generate_answer
from src.selfcheck import extract_claims, check_claims_and_rewrite

class State(TypedDict, total=False):
    question: str
    mode: Literal["chat", "detector"]
    history: List[Dict[str, str]]
    intent: Literal["qa", "detector", "blocked"]
    evidences: List[Dict[str, Any]]
    draft: str
    final: str
    sensitive_category: Optional[str]

SIM_THRESHOLD = 0.50


# =========================
# Guardrails (sensível)
# =========================
GUARDRAILS_ENABLE = os.getenv("GUARDRAILS_ENABLE", "true").lower() == "true"

def _norm(txt: str) -> str:
    txt = (txt or "").lower().strip()
    txt = unicodedata.normalize("NFKD", txt)
    return "".join(ch for ch in txt if not unicodedata.combining(ch))

SENSITIVE_PATTERNS: Dict[str, re.Pattern] = {
    # Armas/explosivos
    "weapons_explosives": re.compile(
        r"\b(bomba(s)?|explosivo(s)?|detonador(es)?|fabricar bomba|improvised explosive|ied|gunpowder|nitrato de amonio)\b"
    ),
    # Autoagressão / suicídio
    "self_harm": re.compile(
        r"\b(suicidio|me matar|tirar a propria vida|autoagress(a|ao)|auto mutila(c|ç)ao|self[- ]?harm|kill myself)\b"
    ),
    # Médico (diagnóstico/prescrição)
    "medical": re.compile(
        r"\b(diagnostico|diagnosticar|prescrever|posologia|dosagem|tratamento|doenc(a|a)|cancer|depress(a|ao)|ansiedade)\b"
    ),
    # Ciberataques/malware
    "cyber_misuse": re.compile(
        r"\b(ddos|phishing|keylogger|ransomware|malware|exploit|zero[- ]?day|bypass auth|crackear|invadir sistema)\b"
    ),
    # Drogas ilícitas
    "illegal_drugs": re.compile(
        r"\b(fabricar (drogas?|metanfetamina|cocaina)|cultivar maconha|extra(c|ç)ao thc|drug lab)\b"
    ),
    # Bio/químico perigoso
    "biohazard": re.compile(
        r"\b(ricina|antraz|agente (biologico|biol[oó]gico)|cultivo de patogeno(s)?|gain of function)\b"
    ),
}

ALLOWLIST = [
    re.compile(r"\bbomba de (chocolate|caloria|gasolina|insulina)\b"),
    re.compile(r"\b(plano de saude|politica de saude ocupacional)\b"),
    re.compile(r"\b(analise etica|debate regulatorio|alto nivel|overview)\b"),
]

def detect_sensitive(text: str) -> (bool, Optional[str]):
    if not GUARDRAILS_ENABLE:
        return False, None
    t = _norm(text)
    for pat in ALLOWLIST:
        if pat.search(t):
            return False, None
    for label, pat in SENSITIVE_PATTERNS.items():
        if pat.search(t):
            return True, label
    return False, None

def refusal_message(category: Optional[str]) -> str:
    if category == "self_harm":
        return (
            "Sinto muito que você esteja passando por isso. "
            "Não posso fornecer esse tipo de orientação. "
            "Se você corre perigo imediato, ligue para serviços de emergência.\n\n"
            "- **CVV (Brasil)**: 188 — 24h, gratuito\n"
            "- **Chat**: https://www.cvv.org.br/\n\n"
            "Se preferir, posso compartilhar recursos de apoio emocional."
        )
    if category == "medical":
        return (
            "Não posso oferecer diagnóstico ou prescrição. "
            "Procure um profissional de saúde qualificado. "
            "Se quiser, trago informações educacionais gerais e fontes oficiais."
        )
    if category in {"weapons_explosives", "biohazard", "cyber_misuse", "illegal_drugs"}:
        return (
            "Não posso ajudar a planejar, instruir ou facilitar atividades perigosas ou ilegais. "
            "Posso, no máximo, discutir riscos, ética e prevenção em termos gerais."
        )
    return (
        "Desculpe, não posso ajudar com esse tipo de solicitação. "
        "Posso oferecer informações de alto nível e recursos confiáveis."
    )

# =========================
# Similaridade util
# =========================
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

# =========================
# Nós do grafo
# =========================
def _router(state: State) -> Dict[str, Any]:
    q = (state.get("question") or "")
    is_sensitive, category = detect_sensitive(q)
    if is_sensitive:
        return {"intent": "blocked", "sensitive_category": category}

    mode = state.get("mode") or "chat"
    if mode == "detector" or any(t in q.lower() for t in ["é golpe", "fraude", "suspeita", "phishing", "link", "sms"]):
        return {"intent": "detector"}
    return {"intent": "qa"}

def _retrieve(state: State, retriever: PDFIndexerRetriever) -> Dict[str, Any]:
    # Se bloqueado por segurança, não consulta índice.
    if state.get("intent") == "blocked":
        return {"evidences": []}

    q = state["question"]
    local_hits = retriever.retrieve(q, k=8)
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
    # Se bloqueado, retorna recusa direta (sem chamar LLM)
    if state.get("intent") == "blocked":
        msg = refusal_message(state.get("sensitive_category"))
        return {"draft": msg}

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
    # Se bloqueado, não tenta citar/reescrever
    if state.get("intent") == "blocked":
        return {"final": state.get("draft") or refusal_message(state.get("sensitive_category"))}

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
    # Já tratamos no router/answer; mantido para extensões futuras (ex.: auditoria)
    return {}

def build_app(retriever: PDFIndexerRetriever):
    g = StateGraph(State)
    g.add_node("router", _router)
    g.add_node("retrieve", lambda s: _retrieve(s, retriever))
    g.add_node("answer", _answer)
    g.add_node("selfcheck", lambda s: _selfcheck(s, retriever))
    g.add_node("safety", _safety)

    g.set_entry_point("router")
    # Mantemos fluxo linear; bloqueio “curto-circuita” via respostas neutras
    g.add_edge("router", "retrieve")
    g.add_edge("retrieve", "answer")
    g.add_edge("answer", "selfcheck")
    g.add_edge("selfcheck", "safety")
    g.add_edge("safety", END)

    return g.compile(checkpointer=MemorySaver())
