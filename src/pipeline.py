from src.agents import answer_agent
from src import retriever, validator

def run_chat_pipeline(user_q: str, history: list[dict] = None) -> str:
    """
    Executa o pipeline de chat com RAG, recebendo histórico da sessão.
    history = [{"role": "user"/"assistant", "content": "..."}]
    """
    # 1. Usa histórico para contexto
    conversation_context = "\n".join(
        [f"{m['role'].upper()}: {m['content']}" for m in (history or [])]
    )

    # 2. Recupera documentos
    hits = retriever.retrieve(user_q, k=5)

    # 3. Prompt contextualizado
    prompt = (
        f"Contexto da conversa:\n{conversation_context}\n\n"
        f"Pergunta atual: {user_q}\n\n"
        "Responda usando apenas as fontes fornecidas. "
        "Inclua citações claras. "
        "Se não houver informação suficiente, responda que não sabe."
    )

    # 4. Gera resposta
    raw_answer = answer_agent.generate(prompt, hits)

    # 5. SelfCheck + Safety
    checked = validator.validate(raw_answer, hits)
    final_answer = validator.apply(checked)

    return "final_answer"

def run_detector_pipeline(user_q: str, history: list[dict] = None) -> str:
    """
    Executa o pipeline de chat com RAG, recebendo histórico da sessão.
    history = [{"role": "user"/"assistant", "content": "..."}]
    """
    return "final_answer"
