# --- sqlite patch for Chroma on Streamlit Cloud (must be first) ---
try:
    import sys, pysqlite3  # provided by pysqlite3-binary
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass
# ------------------------------------------------------------------

import os
import time
import streamlit as st
from huggingface_hub import InferenceClient
import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

# ----------------------------
# Config
# ----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # small, fast, good
LLM_MODEL   = "Qwen/Qwen2-1.5B-Instruct"                 # tiny instruct model
TOP_K = 4
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

# ----------------------------
# Simple splitter
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

# ----------------------------
# HF embedding wrapper
# ----------------------------
class HFEmbeddingFunction:
    def __init__(self, model_id, token=None):
        self.client = InferenceClient(model=model_id, token=token)

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for t in texts:
            # feature_extraction may return a single vector or token-level vectors
            out = self.client.feature_extraction(t)
            if isinstance(out, list) and out and isinstance(out[0], list):
                # average pool token vectors
                dim = len(out[0])
                pooled = [0.0] * dim
                for token_vec in out:
                    for i, val in enumerate(token_vec):
                        pooled[i] += val
                vectors.append([v / len(out) for v in pooled])
            else:
                vectors.append(out)
        return vectors

# ----------------------------
# Read PDF to text
# ----------------------------
def pdf_to_text(file):
    reader = PdfReader(file)
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n\n".join(pages)

# ----------------------------
# Prompt
# ----------------------------
SYS_PROMPT = (
    "You are a concise assistant. Answer the user's question using ONLY the provided context. "
    "If the answer isn't in the context, say you don't know."
)

def build_prompt(context, question):
    return (
        f"{SYS_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

def _use_text2text(model_id: str) -> bool:
    m = model_id.lower()
    return ("t5" in m) or ("flan" in m) or ("mt0" in m) or ("bart" in m)

# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="PDF RAG (HF Free)", page_icon="📄")
st.title("📄 Chat with your PDF — Hugging Face (Free)")

if not HF_TOKEN:
    st.warning("Add your Hugging Face token as HF_TOKEN in Streamlit Secrets.")
    st.stop()

# In-memory Chroma
if "client" not in st.session_state:
    st.session_state.client = chromadb.Client(Settings(anonymized_telemetry=False))
if "collection" not in st.session_state:
    st.session_state.collection = None

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    emb_model_id = st.text_input("Embedding model", EMBED_MODEL)
with col2:
    llm_model_id = st.text_input("LLM model", LLM_MODEL)

if uploaded and st.button("Index document"):
    with st.spinner("Reading and indexing..."):
        text = pdf_to_text(uploaded)
        chunks = chunk_text(text)
        hf_embed = HFEmbeddingFunction(emb_model_id, token=HF_TOKEN)
        coll_name = f"pdf_{int(time.time())}"
        collection = st.session_state.client.create_collection(
            name=coll_name,
            embedding_function=hf_embed
        )
        ids = [f"c_{i}" for i in range(len(chunks))]
        collection.add(documents=chunks, ids=ids)
        st.session_state.collection = collection
        st.success(f"Indexed {len(chunks)} chunks.")

query = st.text_input("Ask a question about the PDF")
go = st.button("Ask")

if go:
    if st.session_state.collection is None:
        st.error("Upload and index a PDF first.")
        st.stop()

    with st.spinner("Retrieving..."):
        res = st.session_state.collection.query(
            query_texts=[query],
            n_results=TOP_K
        )
        contexts = res["documents"][0] if res and res.get("documents") else []
        context_text = "\n\n---\n\n".join(contexts) if contexts else ""

    if not context_text.strip():
        st.info("Couldn't retrieve relevant context. Try a different question.")
    else:
        prompt = build_prompt(context_text, query)
        client = InferenceClient(model=llm_model_id, token=HF_TOKEN)
        with st.spinner("Generating answer (HF inference)..."):
            if _use_text2text(llm_model_id):
                answer = client.text2text_generation(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS
                )
            else:
                answer = client.text_generation(
                    prompt,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    do_sample=True
                )
        st.markdown("### Answer")
        st.write((answer or "").strip())

        with st.expander("Show retrieved context"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i}**\n\n{c}")
