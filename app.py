import streamlit as st
import os
from models.ml_predict import predict_stunting
from rag import (
    get_ai_response,
    load_knowledge_base,
    build_faiss_index,
    search_relevant_chunks,
)

st.set_page_config(page_title="Nutri-Sight", layout="wide")

# Custom CSS untuk UI improvements
st.markdown(
    """
    <style>
    /* Chat container dengan scroll */
    .chat-container {
        height: 500px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
        margin-bottom: 10px;
    }
    .chat-message {
        margin: 10px 0;
        padding: 12px;
        border-radius: 10px;
        max-width: 85%;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
    }
    .bot-message {
        background-color: #ffffff;
        margin-right: auto;
        border: 1px solid #ddd;
    }
    .logo-header {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# Cek apakah API Key sudah diatur
if not GROQ_API_KEY:
    st.error(
        "⚠️ API Key Groq belum diatur. Silakan tambahkan di Streamlit Cloud Settings → Secrets"
    )
    st.stop()


# Load Knowledge Base
@st.cache_resource
def load_system():
    chunks = load_knowledge_base("knowledge_base")
    if chunks:
        index = build_faiss_index(chunks)
        return index, chunks
    return None, []


faiss_index, knowledge_chunks = load_system()

# Initialize session state
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "child_data" not in st.session_state:
    st.session_state.child_data = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header dengan logo
st.markdown('<div class="logo-header">', unsafe_allow_html=True)
if os.path.exists("Nutri-Sight.png"):
    st.image("Nutri-Sight.png", width=60)
st.title("Nutri-Sight: Intervensi Gizi & Stunting")
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("Sistem hibrida Machine Learning dan RAG Chatbot")

# Sidebar untuk info dan kontrol
with st.sidebar:
    st.header("ℹ️ Informasi Sistem")
    st.write("**Model ML:** Random Forest")
    st.write("**Akurasi:** 87.66%")
    st.write("**AI Model:** Llama 3.1 8B")
    st.write("**Knowledge Base:** 4 Guidebook")

    st.markdown("---")

    # Tombol reset
    if st.button("🔄 Reset Prediksi", use_container_width=True):
        st.session_state.prediction_result = None
        st.session_state.child_data = None
        st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

col1, col2 = st.columns([1, 1.5])

# Kolom Input Data Anaak (Antropometri) + Prediksi ML
with col1:
    st.subheader(" Input Data Anak (Antropometri)")

    with st.form("input_form"):
        usia = st.number_input("Usia (Bulan)", min_value=0, max_value=60, value=12)
        berat = st.number_input(
            "Berat Badan (kg)", min_value=0.0, max_value=30.0, value=9.0, step=0.1
        )
        tinggi = st.number_input(
            "Tinggi Badan (cm)", min_value=0.0, max_value=120.0, value=75.0, step=0.1
        )
        gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])

        submit = st.form_submit_button(
            "Prediksi Status Gizi", use_container_width=True
        )

        if submit:
            with st.spinner("Sedang memproses prediksi..."):
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

                    # Tampilkan logo di hasil prediksi
                    if os.path.exists("Nutrisight.png"):
                        st.image("Nutrisight.png", width=40)

                    # Tampilkan hasil 
                    status_emoji = {
                        "Normal": "✅",
                        "Stunted": "⚠️",
                        "Severely Stunted": "🚨",
                        "Tall": "📏",
                    }
                    emoji = status_emoji.get(result["status"], "❓")

                    st.info(
                        f"**{emoji} Status:** {result['status']}\n\n"
                        f"**🎯 Confidence:** {result['confidence']:.2%}"
                    )

                    with st.expander("📈 Detail Perhitungan"):
                        st.write(f"- **BMI:** {result['details']['bmi']} kg/m²")
                        st.write(
                            f"- **Z-Score Tinggi:** {result['details']['z_score']}"
                        )
                        st.write(
                            f"- **Tinggi per Usia:** {result['details']['tinggi_per_usia']} cm/bulan"
                        )
                else:
                    st.error(f"❌ {result['message']}")

    # Tampilkan hasil prediksi jika ada
    if (
        st.session_state.prediction_result
        and st.session_state.prediction_result["success"]
    ):
        st.markdown("---")
        st.markdown("### 📋 Hasil Prediksi Tersimpan")
        pred = st.session_state.prediction_result
        status_emoji = {
            "Normal": "✅",
            "Stunted": "⚠️",
            "Severely Stunted": "🚨",
            "Tall": "📏",
        }
        emoji = status_emoji.get(pred["status"], "❓")

        st.write(f"**{emoji} Status:** {pred['status']}")
        st.write(f"**🎯 Confidence:** {pred['confidence']:.2%}")

# Chatbot
with col2:
    st.subheader("Chatbot Intervensi Gizi & Stunting")

    # Container scrollable untuk chat
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    # Tampilkan chat history dengan styling custom
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(
                f'<div class="chat-message user-message"><b>Anda:</b><br>{message["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            # Tampilkan logo bot di response
            logo_html = ""
            if os.path.exists("Nutri-Sight.png"):
                logo_html = '<img src="Nutri-Sight.png" width="30" style="vertical-align: middle; margin-right: 8px;">'
            st.markdown(
                f'<div class="chat-message bot-message">{logo_html}<b>Nutri-Sight:</b><br>{message["content"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    # Input chat (tetap di bawah, tidak ikut scroll)
    if prompt := st.chat_input(
        "Tanyakan tentang gizi, stunting, atau hasil prediksi..."
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Buat konteks dari hasil prediksi (jika ada)
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

Berikan rekomendasi yang relevan dengan status gizi anak ini berdasarkan dokumen referensi.
"""
                else:
                    konteks_anak = "Tidak ada data prediksi. Berikan informasi umum berdasarkan pertanyaan user."

                # Search relevant chunks
                if faiss_index and knowledge_chunks:
                    relevant_chunks = search_relevant_chunks(
                        prompt, faiss_index, knowledge_chunks, top_k=2
                    )
                else:
                    relevant_chunks = []

                # Get AI response
                response = get_ai_response(
                    question=prompt,
                    child_context=konteks_anak,
                    relevant_chunks=relevant_chunks,
                    api_key=GROQ_API_KEY,
                )

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun()  # Refresh untuk update chat container
def build_faiss_index(texts):
    """Bangun FAISS index dari list teks"""
    if not texts:
        return None
