# Assistente Anti-Fraude

PoC open-source de assistente RAG + agentes para orientar cidadãos sobre prevenção e reação a fraudes, com respostas ancoradas em fontes oficiais (Banco Central).

## Setup (mínimo)

1. Clonar:
   git clone <repo-url>
2. Criar venv:
   python3 -m venv .venv && source .venv/bin/activate
3. Instalar:
   python3 -m pip install --no-cache --no-deps -r requirements.txt
4. Ingestão:
   python3 src/retriever.py
5. Rodar app:
   PYTHONPATH=. streamlit run app/streamlit_app.py

## Execução local com Ollama (opcional)

- Instale Ollama (https://ollama.com)
- Baixe modelo (ex.: llama-3.1-8b)
- export OLLAMA_MODEL=llama-3.1-8b

## Avaliação para cada métrica

1. PYTHONPATH=. python eval/eval_ragas.py --data eval/eval_questions.jsonl --mode chat --k 8 --metric faithfulness_hhem --outdir reports
2. PYTHONPATH=. python eval/eval_ragas.py --data eval/eval_questions.jsonl --mode chat --k 8 --metric answer_relevancy --outdir reports
