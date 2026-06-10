import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from groq import Groq

# Model embedding (Bahasa Indonesia)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
print("✅ Embedding model loaded!")


def extract_text_from_pdf(pdf_path):
    """Extract text dari file PDF"""
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                extracted = page.extract_text()
                if extracted:
                    text += extracted + " "
    except Exception as e:
        print(f"️ Error membaca {pdf_path}: {e}")
    return text


def load_knowledge_base(pdf_folder="knowledge_base"):
    """Load semua PDF dari folder knowledge_base"""
    chunks = []

    if not os.path.exists(pdf_folder):
        print(f"️ Folder '{pdf_folder}' tidak ditemukan!")
        os.makedirs(pdf_folder, exist_ok=True)
        return chunks

    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    if not pdf_files:
        print(f"️ Tidak ada file PDF di folder '{pdf_folder}'!")
        return chunks

    print(f" Loading {len(pdf_files)} file PDF...")

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"  - Processing: {filename}")

        text = extract_text_from_pdf(pdf_path)

        if not text.strip():
            print(f"    ⚠️ File kosong atau tidak ada teks")
            continue

        # Chunk size kecil untuk menghindari limit token Groq (413 Error)
        chunk_size = 300
        overlap = 50

        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i : i + chunk_size]
            if chunk_text.strip():
                chunks.append({"content": chunk_text.strip(), "source": filename})

    print(f"✅ Total {len(chunks)} chunks berhasil dibuat!")
    return chunks


def build_faiss_index(knowledge_chunks):
    """Build FAISS index dari chunks"""
    if not knowledge_chunks:
        print("⚠️ Tidak ada chunks untuk di-index!")
        return None

    print("🔄 Building FAISS index...")

    texts = [chunk["content"] for chunk in knowledge_chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"✅ FAISS index berhasil dibuat! ({len(texts)} vectors)")
    return index


def search_relevant_chunks(question, index, knowledge_chunks, top_k=4):
    """Search chunks yang paling relevan"""
    if index is None or not knowledge_chunks:
        return []

    q_embedding = embedding_model.encode(
        [question], convert_to_numpy=True, normalize_embeddings=True
    )

    # Ambil top_k chunks
    distances, indices = index.search(q_embedding, min(top_k, len(knowledge_chunks)))

    relevant_chunks = []
    for idx in indices[0]:
        if idx < len(knowledge_chunks):
            relevant_chunks.append(knowledge_chunks[idx])

    return relevant_chunks


def format_context(relevant_chunks):
    """Format chunks menjadi context string yang ringkas"""
    if not relevant_chunks:
        return "Tidak ada dokumen referensi."

    context = ""
    for i, chunk in enumerate(relevant_chunks, 1):
        # Batasi panjang konten per chunk agar tidak kena limit token
        content = chunk["content"][:150]
        context += f"\n[{i}] {content}...\n"

    return context


def get_ai_response(question, child_context, relevant_chunks, api_key):
    """Get response dari Groq API dengan System Prompt yang SANGAT KETAT"""

    knowledge_snippet = format_context(relevant_chunks)

    # Batasi panjang child_context
    if len(child_context) > 200:
        child_context = child_context[:200] + "..."

    # System Prompt yang sangat ketat untuk menjaga konteks
    system_prompt = f"""Anda adalah Nutri-Sight, asisten cerdas untuk intervensi gizi dan kesehatan anak.

ATURAN WAJIB:
1. Anda HANYA boleh menjawab pertanyaan seputar: gizi anak, stunting, kesehatan balita, nutrisi, ASI, MPASI, imunisasi, pertumbuhan anak, intervensi gizi, pencegahan stunting, posyandu, KIA, dan topik terkait kesehatan & gizi anak yang ada di DOKUMEN REFERENSI.
2. Jika pertanyaan TIDAK terkait konteks di atas (misal: coding, matematika, sapaan umum seperti 'halo' tanpa konteks, atau 'buatkan kode'), jawab PERSIS: "Maaf, saya hanya dapat membantu pertanyaan seputar gizi dan kesehatan anak berdasarkan dataset yang tersedia."
3. Jika ditanya "kamu siapa" atau sejenisnya, jawab PERSIS: "Saya Nutri-Sight, asisten cerdas untuk intervensi seputar kesehatan dan gizi anak."
4. Jika pertanyaan tidak jelas (seperti "tau nggak", "bisa gak"), tanya balik: "Tau tentang apa? Silakan tanyakan seputar gizi dan kesehatan anak ya 😊"
5. JANGAN PERNAH membuat kode programming, rumus matematika, atau hal teknis di luar konteks gizi.
6. Jawab HANYA berdasarkan DOKUMEN REFERENSI di bawah.
7. Gunakan Bahasa Indonesia yang ramah dan mudah dipahami.

KONTEKS DATA ANAK:
{child_context}

DOKUMEN REFERENSI:
{knowledge_snippet}
"""

    try:
        client = Groq(api_key=api_key)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": question[:200],
                },  # Batasi panjang pertanyaan user
            ],
            temperature=0.2,
            max_tokens=400,  # Batasi output agar tidak kena limit
            top_p=0.9,
        )

        response_text = response.choices[0].message.content.strip()

        # Hapus bagian <think> jika ada (fallback jika model reasoning digunakan)
        if "<think>" in response_text and "</think>" in response_text:
            response_text = response_text.split("</think>")[-1].strip()
        elif "<think>" in response_text:
            start_idx = response_text.find("</think>")
            if start_idx != -1:
                response_text = response_text[start_idx + len("</think>") :].strip()

        return response_text

    except Exception as e:
        error_msg = str(e)
        if "413" in error_msg or "too large" in error_msg.lower():
            return "️ Maaf, sistem sedang overload. Silakan coba pertanyaan yang lebih singkat."
        return f"Error: {str(e)[:100]}"
