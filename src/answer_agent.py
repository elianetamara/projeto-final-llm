import ollama
import os
from src.prompts import get_chat_prompt, get_detector_prompt


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


def generate(user_query: str, hits: list[dict], history: list[dict], prompt_type: str = "chat") -> str:
    evidence_text = "\n".join(
        [f"- {h['text']} (Fonte: {h['meta'].get('source', 'pdf')})" for h in hits]
    ) or "Nenhuma evidência encontrada."

    prompt_template = get_chat_prompt() if prompt_type == "chat" else get_detector_prompt()

    prompt = prompt_template.format(
        query=user_query,
        local_context=evidence_text,
        history=history,
    )

    resp = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, options={"num_predict": 256})
    text = resp['response']
    return text
