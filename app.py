import os
import time
import numpy as np
import requests
import streamlit as st
from huggingface_hub import InferenceClient
from pypdf import PdfReader
from openai import OpenAI

# ========= Config =========
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # embeddings via HF serverless
TOP_K = 4
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

# LLM providers
PROVIDERS = ["OpenAI (gpt-4o-mini)", "HuggingFace serverless (t5-small)"]
DEFAULT_PROVIDER = PROVIDERS[0]
HF_FALLBACK_MODEL = "t5-small"

# ========= Small helpers =========
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
    # Works well for both OpenAI and T5-family
    return (
        "You are a concise assistant. Answer using ONLY the provided context. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

# ========= Embeddings (HF serverless) =========
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

# ========= Generation: OpenAI (reliable) =========
def openai_generate(prompt: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set in secrets")
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a concise assistant. Be accurate and cite only from provided context."},
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )
    return resp.choices[0].message.content.strip()

# ========= Generation: HF serverless fallback (t5-small) =========
def _http_post(url, payload):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for _ in range(5):
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
            raise FileNotFoundError("HF serverless: model not available")
        r.raise_for_status()
        return r.json()
    raise RuntimeError("HF serverless retry limit exceeded")

def hf_t5_generate(model_id: str, prompt: str) -> str:
    payload = {"model": model_id, "inputs": prompt, "parameters": {"max_new_tokens": MAX_NEW_TOKENS}}
    out = _http_post("https://api-inference.huggingface.co/pipeline/text2text-generation", payload)
    # normalize
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return (out[0].get("generated_text") or str(out[0])).strip()
    if isinstance(out, dict):
        return (out.get("generated_text") or str(out)).strip()
    return str(out).strip()

# ========= Streamlit UI =========
st.set_page_config(page_title="PDF RAG (HF+OpenAI)", page_icon="📄")
st.title("📄 Chat with your PDF — HF embeddings + OpenAI gen (free-ish)")

if not HF_TOKEN:
    st.warning("Add your Hugging Face token as HF_TOKEN in Streamlit Secrets.")
    st.stop()

if "docs" not in st.session_state:
    st.session_state.docs = []
if "embeds" not in st.session_state:
    st.session_state.embeds = None
if "embed_model_id" not in st.session_state:
    st.session_state.embed_model_id = EMBED_MODEL
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

col1, col2 = st.columns(2)
with col1:
    emb_model_id = st.text_input("Embedding model", st.session_state.embed_model_id)
with col2:
    provider = st.selectbox("LLM provider", PROVIDERS, index=PROVIDERS.index(DEFAULT_PROVIDER))

# Auto-index on upload or embedding change
def _auto_index():
    with st.spinner("Reading and indexing..."):
        text = pdf_to_text(uploaded)
        chunks = chunk_text(text)
        embedder = HFEmbedder(emb_model_id, token=HF_TOKEN)
        vectors = embedder.encode(chunks)
        st.session_state.docs = chunks
        st.session_state.embeds = vectors
        st.session_state.embed_model_id = emb_model_id
        st.session_state.last_file_name = uploaded.name
        st.success(f"Indexed {len(chunks)} chunks.")

if uploaded is not None:
    if (
        st.session_state.last_file_name != uploaded.name
        or st.session_state.embed_model_id != emb_model_id
        or st.session_state.embeds is None
    ):
        _auto_index()

query = st.text_input("Ask a question about the PDF")
go = st.button("Ask")

if go:
    if st.session_state.embeds is None or len(st.session_state.docs) == 0:
        st.error("Upload a PDF first.")
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
            if provider.startswith("OpenAI"):
                answer = openai_generate(prompt)
                used = "OpenAI gpt-4o-mini"
            else:
                # HF serverless fallback (free); use a safe small model
                answer = hf_t5_generate(HF_FALLBACK_MODEL, prompt)
                used = f"HF serverless ({HF_FALLBACK_MODEL})"
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

        st.markdown("### Answer")
        st.write((answer or "").strip())
        st.caption(f"Generator: {used}")

        with st.expander("Show retrieved context"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i} (score: {float(sims[i-1]):.3f})**\n\n{c}")
