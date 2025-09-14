import re

def assertive_sentences(text):
    # heurística simples: quebra por sentenças e filtra as que parecem assertivas (contêm verbo no indicativo)
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sents if len(s.strip())>0]

def check_coverage(answer_text, evidence_list):
    # Regra simples: toda sentença que contém números, datas, ações ou "deve/é recomendado" precisa referência [fonte i]
    sents = assertive_sentences(answer_text)
    problems = []
    for s in sents:
        if any(tok in s.lower() for tok in ["deve", "é recomendado", "recomenda", "será", "pode", "é obrigatório", "é necessário", "proceda", "solicitar", "como solicitar"]):
            if "[fonte" not in s and "fonte" not in s:
                problems.append({"sentence": s, "issue":"sem citação"})
    return problems
