import os
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ----------------------------
# Config
# ----------------------------
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"   # ~80MB, good quality
GEN_MODEL_ID   = "google/flan-t5-small"                     # ~300MB, CPU-friendly
TOP_K = 4
MAX_NEW_TOKENS = 256

# ----------------------------
# Helpers
# ----------------------------
def chunk_text(text, chunk_size=900, overlap=150):
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks

def pdf_to_text(file):
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

def t5_prompt(context, question):
    return (
        "Given the following context, answer the question concisely.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

def normalize_rows(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    return matrix / norms

# ----------------------------
# Lazy-load models once
# ----------------------------
@st.cache_resource(show_spinner="Loading models… this happens only once")
def load_models():
    embedder = SentenceTransformer(EMBED_MODEL_ID, device="cpu")
    # flan-t5-small text2text pipeline
    generator = pipeline("text2text-generation", model=GEN_MODEL_ID, device=-1)
    return embedder, generator

embedder, generator = load_models()

# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="PDF RAG (HF-only, local)", page_icon="📄")
st.title("📄 Chat with your PDF — Hugging Face only (no APIs)")

if "docs" not in st.session_state:
    st.session_state.docs = []
if "embeds" not in st.session_state:
    st.session_state.embeds = None
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

# Auto-index right after upload (or if a different file is uploaded)
def auto_index():
    with st.spinner("Reading and indexing…"):
        text = pdf_to_text(uploaded)
        chunks = chunk_text(text)
        # Sentence-Transformers encode -> numpy, normalized for cosine
        vecs = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
        vecs = normalize_rows(vecs)
        st.session_state.docs = chunks
        st.session_state.embeds = vecs
        st.session_state.last_file_name = uploaded.name
        st.success(f"Indexed {len(chunks)} chunks.")

if uploaded is not None:
    if st.session_state.last_file_name != uploaded.name or st.session_state.embeds is None:
        auto_index()

query = st.text_input("Ask a question about the PDF")
if st.button("Ask"):
    if st.session_state.embeds is None or len(st.session_state.docs) == 0:
        st.error("Upload a PDF first.")
        st.stop()

    # Retrieve
    q = embedder.encode([query], convert_to_numpy=True, show_progress_bar=False)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    sims = (st.session_state.embeds @ q.T).ravel()
    if sims.size == 0:
        st.info("Couldn't retrieve relevant context. Try a different question.")
        st.stop()

    topk_idx = np.argsort(-sims)[:TOP_K]
    contexts = [st.session_state.docs[i] for i in topk_idx]
    context_text = "\n\n---\n\n".join(contexts)

    # Generate
    prompt = t5_prompt(context_text, query)
    out = generator(prompt, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    answer = (out[0]["generated_text"] if isinstance(out, list) else str(out)).strip()

    st.markdown("### Answer")
    st.write(answer)
    with st.expander("Show retrieved context"):
        for i, c in enumerate(contexts, 1):
            st.markdown(f"**Chunk {i} (score: {float(sims[topk_idx[i-1]]):.3f})**\n\n{c}")
