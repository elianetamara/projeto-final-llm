import re


def assertive_sentences(text: str) -> list[str]:
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if s.strip()]


def check_coverage(answer_text: str) -> list[dict[str, str]]:
    sents = assertive_sentences(answer_text)
    problems = []
    for s in sents:
        if any(tok in s.lower() for tok in [
            "deve", "é recomendado", "recomenda", "será", "pode",
            "é obrigatório", "é necessário", "proceda", "solicitar",
            "como solicitar"
        ]):
            if "[fonte" not in s and "fonte" not in s:
                problems.append({"sentence": s, "issue": "sem citação"})
    return problems


def add_disclaimer(answer_text: str) -> str:
    disclaimer = (
        "\n\nAviso: Esta resposta é informativa. Não constitui aconselhamento legal ou médico. "
        "Para recuperar valores perdidos, contate imediatamente sua instituição financeira e a polícia. "
    )
    return answer_text + disclaimer
