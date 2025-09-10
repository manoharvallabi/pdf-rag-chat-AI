import os
import time
import json
import numpy as np
import requests
import streamlit as st
from huggingface_hub import InferenceClient
from pypdf import PdfReader

# ----------------------------
# Config
# ----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # embeddings via serverless
# Use a known-working free model by default; not all google/* are on serverless.
LLM_MODEL   = "t5-small"
TOP_K = 4
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

# If a model 404s, we try these in order automatically.
LLM_FALLBACKS = [
    "t5-small",
    "google/flan-t5-small",
    "bigscience/mt0-small",
    "facebook/bart-base",
]

# ----------------------------
# Utils
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

def build_prompt(context, question):
    SYS_PROMPT = (
        "You are a concise assistant. Answer the user's question using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know."
    )
    return (
        f"{SYS_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

# ----------------------------
# HF helpers
# ----------------------------
class HFEmbedder:
    def __init__(self, model_id: str, token: str | None):
        self.client = InferenceClient(model=model_id, token=token)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            out = self.client.feature_extraction(t)
            # If token-level vectors are returned, mean-pool them
            if isinstance(out, list) and out and isinstance(out[0], list):
                arr = np.array(out, dtype=np.float32)  # (tokens, dim)
                vec = arr.mean(axis=0)
            else:
                vec = np.array(out, dtype=np.float32)
            vecs.append(vec)
        return np.vstack(vecs)

def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, k: int):
    def _norm(x):
        n = np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9
        return x / n
    q = _norm(query_vec.reshape(1, -1))  # (1, d)
    M = _norm(matrix)                    # (n, d)
    sims = (M @ q.T).ravel()             # (n,)
    if len(sims) == 0:
        return np.array([], dtype=int), np.array([])
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

# ----------------------------
# Serverless generation via HTTP (robust)
# ----------------------------
def hf_generate_http(model_id: str, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Call Hugging Face serverless inference endpoint directly.
    Works for text-generation and text2text-generation models.
    Handles 503 "loading" with short retries.
    """
    url = f"https://api-inference.huggingface.co/models/{model_id}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": max_new_tokens}}

    for attempt in range(5):
        r = requests.post(url, headers=headers, json=payload, timeout=120)
        if r.status_code == 503:
            # model warming up
            try:
                wait = float(r.json().get("estimated_time", 5))
            except Exception:
                wait = 5
            time.sleep(min(20, wait + 1))
            continue
        if r.status_code == 404:
            raise FileNotFoundError(f"Model not available on serverless: {model_id}")
        r.raise_for_status()

        out = r.json()
        # Normalize common shapes
        if isinstance(out, list) and out and isinstance(out[0], dict):
            return out[0].get("generated_text") or out[0].get("translation_text") or out[0].get("summary_text") or str(out[0])
        if isinstance(out, dict):
            return out.get("generated_text") or out.get("answer") or out.get("translation_text") or out.get("summary_text") or str(out)
        return str(out)

    raise RuntimeError("HF serverless generation retry limit exceeded")

def generate_with_fallback(model_id: str, prompt: str):
    # Try requested model, then fallbacks if 404/other issues
    candidates = [model_id] + [m for m in LLM_FALLBACKS if m != model_id]
    last_err = None
    for mid in candidates:
        try:
            return hf_generate_http(mid, prompt, MAX_NEW_TOKENS), mid
        except FileNotFoundError:
            last_err = f"{mid} not on serverless"
            continue
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"All serverless attempts failed. Last error: {last_err}")

# ----------------------------
# App
# ----------------------------
st.set_page_config(page_title="PDF RAG (HF Free, Lite)", page_icon="📄")
st.title("📄 Chat with your PDF — Hugging Face (Lite, no sqlite)")

if not HF_TOKEN:
    st.warning("Add your Hugging Face token as HF_TOKEN in Streamlit Secrets.")
    st.stop()

if "docs" not in st.session_state:
    st.session_state.docs = []
if "embeds" not in st.session_state:
    st.session_state.embeds = None
if "embed_model_id" not in st.session_state:
    st.session_state.embed_model_id = EMBED_MODEL

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    emb_model_id = st.text_input("Embedding model", st.session_state.embed_model_id)
with col2:
    llm_model_id = st.text_input("LLM model", LLM_MODEL)

if uploaded and st.button("Index document"):
    with st.spinner("Reading and indexing..."):
        text = pdf_to_text(uploaded)
        chunks = chunk_text(text)
        embedder = HFEmbedder(emb_model_id, token=HF_TOKEN)
        vectors = embedder.encode(chunks)
        st.session_state.docs = chunks
        st.session_state.embeds = vectors
        st.session_state.embed_model_id = emb_model_id
        st.success(f"Indexed {len(chunks)} chunks.")

query = st.text_input("Ask a question about the PDF")
go = st.button("Ask")

if go:
    if st.session_state.embeds is None or len(st.session_state.docs) == 0:
        st.error("Upload and index a PDF first.")
        st.stop()

    with st.spinner("Retrieving..."):
        embedder = HFEmbedder(st.session_state.embed_model_id, token=HF_TOKEN)
        qvec = embedder.encode(query)  # (1, d)
        idx, sims = cosine_topk(qvec[0], st.session_state.embeds, TOP_K)
        contexts = [st.session_state.docs[i] for i in idx]
        context_text = "\n\n---\n\n".join(contexts)

    if not context_text.strip():
        st.info("Couldn't retrieve relevant context. Try a different question.")
    else:
        prompt = build_prompt(context_text, query)
        try:
            answer, used_model = generate_with_fallback(llm_model_id.strip(), prompt)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

        st.markdown("### Answer")
        st.write((answer or "").strip())
        st.caption(f"Model: {used_model}")

        with st.expander("Show retrieved context"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i} (score: {float(sims[i-1]):.3f})**\n\n{c}")
