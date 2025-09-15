import ollama
import os

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama-3.1-8b")

def build_prompt(question, evidence_list):
    # evidence_list: list of dict {text, meta}
    prompt = "Você é um assistente que responde sobre prevenção de golpes PIX. Use SOMENTE as evidências fornecidas.\n\n"
    prompt += "EVIDÊNCIAS:\n"
    for i,e in enumerate(evidence_list):
        src = e['meta'].get('source','unknown')
        page = e['meta'].get('page', None)
        prompt += f"[{i}] Fonte: {src} Página:{page}\n{e['text'][:1000].strip()}\n---\n"
    prompt += f"\nPERGUNTA: {question}\n\nInstruções: Responda em português, inclua citações do tipo [fonte i] referenciando as evidências acima. Se não houver evidência para alguma afirmação, diga 'Não encontrado nas fontes'. Não forneça conselho legal ou diagnóstico.\n"
    return prompt

def generate(question, evidence):
    prompt = build_prompt(question, evidence)
    # call to LLM (Ollama or other)
    resp = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, max_tokens=512)
    text = resp['text']
    # Retornar texto + citações (parsing simples)
    return text
