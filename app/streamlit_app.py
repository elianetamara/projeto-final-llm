import streamlit as st
from src.pipeline import run_pipeline

st.set_page_config(
    page_title="Assistente Anti-Fraude",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Assistente Anti-Fraude")
st.markdown(
    "Este assistente ajuda a entender **mecanismos de segurança** "
    "e a **identificar mensagens suspeitas de golpes**.\n\n"
    "**Aviso**: Este sistema é apenas informativo e não substitui a "
    "orientação oficial do seu banco ou autoridades."
)

tab1, tab2 = st.tabs(["💬 Chat Mecanismos de Segurança", "🔍 Detector de Golpes"])

with tab1:
    st.subheader("💬 Tire suas dúvidas sobre segurança digital")
    st.markdown(
        "Pergunte sobre Pix, MED, autenticação dupla, etc")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_q = st.chat_input("Digite sua pergunta...")
    if user_q:
        st.chat_message("user").markdown(user_q)
        st.session_state.chat_history.append(
            {"role": "user", "content": user_q})

        with st.chat_message("assistant"):
            with st.spinner("Buscando informações..."):
                try:
                    answer = run_pipeline(
                        "chat", user_q, history=st.session_state.chat_history)
                except Exception as e:
                    answer = f"⚠️ Erro ao processar pergunta: {e}"

            st.markdown(answer)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": answer})

with tab2:
    st.subheader("🔍 Analisador de mensagens suspeitas")
    st.markdown(
        "Cole aqui a mensagem ou link recebido por SMS/WhatsApp para análise."
    )

    user_msg = st.text_area("Mensagem suspeita:")
    if st.button("Analisar", key="detector_button") and user_msg:
        with st.spinner("Analisando mensagem..."):
            try:
                analysis = run_pipeline("detector", user_msg)
                st.markdown("### ⚠️ Resultado da análise")
                st.write(analysis)
            except Exception as e:
                st.error(f"Erro ao analisar mensagem: {e}")
