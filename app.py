import os
import numpy as np
import streamlit as st
from huggingface_hub import InferenceClient
from pypdf import PdfReader

# ----------------------------
# Config
# ----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL   = "google/flan-t5-base"   # works on HF serverless
TOP_K = 4
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2

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

def _unwrap(resp):
    # HF can return str, dict, or list[dict]
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        return resp.get("generated_text") or str(resp)
    if isinstance(resp, list) and resp and isinstance(resp[0], dict):
        return resp[0].get("generated_text") or str(resp[0])
    return str(resp)

def generate_answer(client: InferenceClient, model_id: str, prompt: str):
    m = model_id.lower()
    # Prefer the text2text path for FLAN/T5/MT0/BART
    if any(t in m for t in ("t5", "flan", "mt0", "bart", "mbart")):
        try:
            # Some hub versions expose this:
            return client.text2text_generation(prompt, max_new_tokens=MAX_NEW_TOKENS)
        except Exception:
            # Generic call to the correct task
            resp = client.post(
                json={"inputs": prompt, "parameters": {"max_new_tokens": MAX_NEW_TOKENS}},
                task="text2text-generation",
            )
            return _unwrap(resp)
    # Fallback: causal LM path
    try:
        return client.text_generation(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=False,
        )
    except Exception:
        resp = client.post(
            json={"inputs": prompt, "parameters": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE, "do_sample": False}},
            task="text-generation",
        )
        return _unwrap(resp)

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
        client = InferenceClient(model=llm_model_id, token=HF_TOKEN)
        try:
            answer = generate_answer(client, llm_model_id, prompt)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

        st.markdown("### Answer")
        st.write((answer or "").strip())

        with st.expander("Show retrieved context"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i} (score: {float(sims[i-1]):.3f})**\n\n{c}")
