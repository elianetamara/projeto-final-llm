from langchain_community.tools.tavily_search import TavilySearchResults

from validator import check_coverage, add_disclaimer
from answer_agent import generate
from src import retriever


def run_pipeline(type: str, user_input: str, history: list[dict] = None) -> str:
    hits = retriever.retrieve(user_input, k=5)
    web_hits = []

    if (type == "detector"):
        web_hits_raw = TavilySearchResults(k=5).run(user_input)
        web_hits = [{"text": r, "meta": {"source": "web"}}
                    for r in web_hits_raw]

    prompt_hits = hits + web_hits if type == "detector" else hits

    raw_answer = generate(user_input, prompt_hits,
                          history, prompt_type="detector")

    problems = check_coverage(raw_answer)
    if problems:
        raw_answer += "\n\n⚠️ Algumas sentenças podem estar sem referência adequada."

    final_answer = add_disclaimer(raw_answer)

    return final_answer
