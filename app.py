import streamlit as st
import os
import sys

# Tambahkan root directory ke Python path agar import module bekerja di Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.ml_predict import predict_stunting
from rag import (
    get_ai_response,
    load_knowledge_base,
    build_faiss_index,
    search_relevant_chunks,
)

st.set_page_config(page_title="Nutri-Sight", layout="wide")

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("⚠️ API Key Groq belum diatur.")
    st.stop()


@st.cache_resource
def load_system():
    chunks = load_knowledge_base("knowledge_base")
    if chunks:
        index = build_faiss_index(chunks)
        return index, chunks
    return None, []


faiss_index, knowledge_chunks = load_system()

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "child_data" not in st.session_state:
    st.session_state.child_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header dengan Logo
col_logo, col_title = st.columns([0.1, 0.9])
with col_logo:
    if os.path.exists("Nutrisight.png"):
        st.image("Nutrisight.png", width=60)
with col_title:
    st.title("NutriSight")
    st.markdown(
        "Chatbot Intervensi Gizi Berbasis RAG dengan Sistem Hibrida Machine Learning untuk Klasifikasi Risiko Stunting"
    )

st.markdown("---")

with st.sidebar:
    st.header("Informasi Sistem")
    st.write("**Model ML:** Random Forest")
    st.write("**Akurasi:** 87.66%")
    st.write("**AI Model:** Llama 3.1 8B")
    st.write("**Knowledge Base:** 3 Guidebook")
    st.markdown("---")
    if st.button("🔄 Reset Prediksi", use_container_width=True):
        st.session_state.prediction_result = None
        st.session_state.child_data = None
        st.rerun()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("Input Data Anak (Antropometri)")

    with st.form("input_form"):
        usia = st.number_input("Usia (Bulan)", min_value=0, max_value=60, value=12)
        berat = st.number_input(
            "Berat Badan (kg)", min_value=0.0, max_value=30.0, value=9.0, step=0.1
        )
        tinggi = st.number_input(
            "Tinggi Badan (cm)", min_value=0.0, max_value=120.0, value=75.0, step=0.1
        )
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

        submit = st.form_submit_button("Prediksi Status Gizi", use_container_width=True)

        if submit:
            with st.spinner("Memproses prediksi..."):
                result = predict_stunting(usia, berat, tinggi, gender)

                if result["success"]:
                    st.session_state.prediction_result = result
                    st.session_state.child_data = {
                        "usia": usia,
                        "berat": berat,
                        "tinggi": tinggi,
                        "gender": gender,
                    }
                    st.success("✅ Prediksi Berhasil!")

                    status_emoji = {
                        "Normal": "✅",
                        "Stunted": "⚠️",
                        "Severely Stunted": "🚨",
                        "Tall": "📏",
                    }
                    emoji = status_emoji.get(result["status"], "❓")

                    st.info(
                        f"**{emoji} Status:** {result['status']}\n\n**Confidence:** {result['confidence']:.2%}"
                    )

                    with st.expander("Detail Perhitungan"):
                        st.write(f"- BMI: {result['details']['bmi']} kg/m²")
                        st.write(f"- Z-Score Tinggi: {result['details']['z_score']}")
                        st.write(
                            f"- Tinggi per Usia: {result['details']['tinggi_per_usia']} cm/bulan"
                        )
                else:
                    st.error(result["message"])

    if (
        st.session_state.prediction_result
        and st.session_state.prediction_result["success"]
    ):
        st.markdown("---")
        st.markdown("### Hasil Prediksi Tersimpan")
        pred = st.session_state.prediction_result
        status_emoji = {
            "Normal": "✅",
            "Stunted": "⚠️",
            "Severely Stunted": "🚨",
            "Tall": "📏",
        }
        emoji = status_emoji.get(pred["status"], "❓")
        st.write(f"**{emoji} Status:** {pred['status']}")
        st.write(f"**Confidence:** {pred['confidence']:.2%}")

with col2:
    st.subheader("Chatbot Intervensi Gizi & Stunting")

    # Chat container dengan scroll
    chat_container = st.container(height=450, border=True)

    with chat_container:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                # User message
                st.markdown(
                    f"""
                    <div style="
                        background-color: #1E3A5F;
                        padding: 12px 16px;
                        border-radius: 12px;
                        margin: 10px 0;
                        margin-left: 20%;
                    ">
                        {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                # Bot message
                st.markdown(
                    f"""
                    <div style="
                        background-color: #262730;
                        padding: 12px 16px;
                        border-radius: 12px;
                        margin: 10px 0;
                        margin-right: 20%;
                    ">
                        {message['content']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # Input chat di luar container
    if prompt := st.chat_input(
        "Tanyakan tentang gizi, stunting, atau hasil prediksi..."
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            if (
                st.session_state.prediction_result
                and st.session_state.prediction_result["success"]
            ):
                pred = st.session_state.prediction_result
                child = st.session_state.child_data

                konteks_anak = f"""
DATA ANAK:
- Usia: {child['usia']} bulan
- Berat Badan: {child['berat']} kg
- Tinggi Badan: {child['tinggi']} cm
- Jenis Kelamin: {child['gender']}

HASIL PREDIKSI ML:
- Status: {pred['status']}
- Confidence: {pred['confidence']:.2%}
- BMI: {pred['details']['bmi']}
- Z-Score Tinggi: {pred['details']['z_score']}
"""
            else:
                konteks_anak = "Tidak ada data prediksi."

            if faiss_index and knowledge_chunks:
                relevant_chunks = search_relevant_chunks(
                    prompt, faiss_index, knowledge_chunks, top_k=2
                )
            else:
                relevant_chunks = []

            response = get_ai_response(
                question=prompt,
                child_context=konteks_anak,
                relevant_chunks=relevant_chunks,
                api_key=GROQ_API_KEY,
            )

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
