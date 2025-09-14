def safety_filter(answer_text):
    # se detectar pedido de diagnóstico ou instruções perigosas -> recusar
    banned_keywords = ["diagnosticar", "medicação", "remédio", "instrução cirúrgica"]
    if any(k in answer_text.lower() for k in banned_keywords):
        return False, "Desculpe, não posso fornecer diagnósticos ou instruções médicas. Procure um profissional."
    return True, answer_text

def add_disclaimer(answer_text):
    disclaimer = "\n\nAviso: Esta resposta é informativa. Não constitui aconselhamento legal ou médico. Para recuperar valores perdidos, contate imediatamente sua instituição financeira e a polícia. Consulte fontes oficiais listadas abaixo."
    return answer_text + disclaimer
