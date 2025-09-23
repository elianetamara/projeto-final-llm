# src/prompts.py
from langchain_core.prompts import ChatPromptTemplate

def get_chat_prompt():
    return ChatPromptTemplate.from_template(r"""
[SISTEMA]
Você é um assistente antifraude. Responda **estritamente** com base no CONTEXTO NUMERADO.
Se não houver base suficiente, responda **exatamente**: "NÃO ENCONTREI BASE".
Não use conhecimento externo.

[HISTÓRICO]
{history}

[PERGUNTA]
{query}

[CONTEXTO NUMERADO]
{local_context}

[REGRAS]
- Cada frase factual deve ter ao menos uma citação no formato [n].
- Se um fato não estiver no CONTEXTO, **não o mencione**.
- Português do Brasil. Sem passo-a-passo sensível.
- Seja conciso.

[SAÍDA]
Texto conciso com citações [n] ao final das frases.
""")

def get_detector_prompt():
    return ChatPromptTemplate.from_template(r"""
[SISTEMA]
Você analisa mensagens suspeitas (Pix/golpes). Use apenas o CONTEXTO NUMERADO.

[HISTÓRICO]
{history}

[MENSAGEM SUSPEITA]
{query}

[FONTES (CONTEXTO NUMERADO)]
{local_context}

[INSTRUÇÕES]
1) Resuma a mensagem.
2) Compare com padrões nas fontes.
3) Diga se há indícios de fraude (Baixo/Médio/Alto) e por quê.
4) Sempre cite fontes [n].
5) Se não houver base suficiente, escreva: "NÃO ENCONTREI BASE".
""")
