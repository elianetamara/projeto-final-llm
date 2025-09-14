import streamlit as st
from src.retriever import Retriever
from src.agents.answer_agent import generate_answer
from src.agents.selfcheck_agent import check_coverage
from src.agents.safety_agent import safety_filter, add_disclaimer

st.set_page_config(page_title="Assistente Anti-Fraude PIX (PoC)")
st.title("Assistente Anti-Fraude PIX — PoC")

q = st.text_area("Pergunta sobre PIX / golpe / link suspeito:", height=120)
k = st.sidebar.slider("Número de evidências (k)", 1, 10, 5)

if st.button("Responder"):
    retr = Retriever()
    hits = retr.retrieve(q, k=k)
    st.subheader("Evidências selecionadas")
    for i,h in enumerate(hits):
        st.markdown(f"**[{i}]** Fonte: {h['meta'].get('source')} — trecho: {h['text'][:400]}...")
        st.write("----")

    answer = generate_answer(q, hits)
    # self-check
    problems = check_coverage(answer, hits)
    ok, filtered = safety_filter(answer)
    if not ok:
        st.warning(filtered)
    else:
        ans_with_disclaimer = add_disclaimer(filtered)
        st.subheader("Resposta gerada")
        st.write(ans_with_disclaimer)
        if problems:
            st.error("SELF-CHECK: Algumas sentenças não possuem citação. Ex.:")
            for p in problems: st.write("-", p['sentence'])
    st.subheader("Fontes (URLs)")
    for h in hits:
        if 'source_url' in h['meta']:
            st.write(h['meta']['source_url'])
