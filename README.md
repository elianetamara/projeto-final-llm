# Assistente Anti-Fraude PIX 

PoC open-source de assistente RAG + agentes para orientar cidadãos sobre prevenção e reação a fraudes PIX, com respostas ancoradas em fontes oficiais (Banco Central, Polícia, PROCON).


## Setup (mínimo)
1. Clonar:
   git clone <repo-url>
2. Criar venv:
   python -m venv .venv && source .venv/bin/activate
3. Instalar:
   pip install -r requirements.txt
4. Ingestão:
   python src/ingest/ingest.py
5. Rodar app:
   streamlit run app/streamlit_app.py

## Execução local com Ollama (opcional)
- Instale Ollama (https://ollama.com)
- Baixe modelo (ex.: llama-3.1-8b)
- export OLLAMA_MODEL=llama-3.1-8b

## Avaliação
python eval/run_eval.py
