import io
import os
import re
import time
import numpy as np
import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from huggingface_hub import InferenceClient
from groq import Groq

# =========================
# Config
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384
TOP_K = 3
MAX_NEW_TOKENS = 128

# =========================
# SVG ICONS
# =========================
USER_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 20 20" fill="none" stroke="#666" stroke-linecap="round">
<circle cx="9.5" cy="5.5" r="3"/>
<path d="M15 16.5v-2c0-3.098-2.495-6-5.5-6c-3.006 0-5.5 2.902-5.5 6v2"/>
</svg>
"""

BOT_SVG = """
<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 2048 2048" fill="#666">
<path d="M640 768h128v128H640V768zm512 0h128v128h-128V768zm469 640q35 0 66 13t54 37t37 55t14 66v469h-128v-469q0-18-12-30t-31-13H299q-18 0-30 12t-13 31v469H128v-469q0-35 13-66t37-54t54-37t67-14h341v-128h-85q-35 0-66-13t-55-37t-36-54t-14-67v-85H256V768h128v-85q0-35 13-66t37-54t54-37t67-14h341V303q-29-17-46-47t-18-64q0-27 10-50t27-40t41-28t50-10q27 0 50 10t40 27t28 41t10 50q0 34-17 64t-47 47v209h341q35 0 66 13t54 37t37 55t14 66v85h128v256h-128v85q0 35-13 66t-37 55t-55 36t-66 14h-85v128h341zM512 1109q0 18 12 30t31 13h810q18 0 30-12t13-31V683q0-18-12-30t-31-13H555q-18 0-30 12t-13 31v426zm256 299h384v-128H768v128z"/>
</svg>
"""

# =========================
# Styles
# =========================
st.markdown("""
<style>
/* put Ask + Clear Chat on same row */
.ask-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 6px;
}
.clear-chat {
    cursor: pointer;
    color: #0073e6;
    font-size: 14px;
    text-decoration: underline;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Helper functions
# =========================
def clean_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def pdf_to_text_chunks(data: bytes) -> list[str]:
    pages = []
    try:
        reader = PdfReader(io.BytesIO(data))
        for p in reader.pages:
            pages.append(clean_text(p.extract_text() or ""))
    except Exception:
        pages = []
    if sum(len(p) for p in pages) < 500:
        try:
            text_all = pdfminer_extract_text(io.BytesIO(data)) or ""
            pages = [clean_text(x) for x in text_all.split("\f")]
        except Exception:
            pass
    return [p for p in pages if p]

def get_embedder():
    return InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)

def embed_texts(texts):
    client = get_embedder()
    arr = []
    for t in texts:
        try:
            out = client.feature_extraction(t)
            arr.append(np.mean(np.asarray(out, dtype=np.float32), axis=0))
        except:
            arr.append(np.zeros((DEFAULT_EMBED_DIM,), dtype=np.float32))
    return np.vstack(arr)

def cosine_topk(q: np.ndarray, M: np.ndarray, k: int):
    if M.size == 0 or q.size == 0:
        return np.array([], dtype=int), np.array([])
    qn = q / (np.linalg.norm(q) + 1e-9)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sims = (Mn @ qn.reshape(-1, 1)).ravel()
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def groq_answer(prompt: str):
    client = Groq(api_key=GROQ_API_KEY)
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Answer using only the given context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=MAX_NEW_TOKENS,
    )
    return resp.choices[0].message.content.strip()

# =========================
# State
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs" not in st.session_state:
    st.session_state.docs = []
if "embeds" not in st.session_state:
    st.session_state.embeds = None

# =========================
# Upload PDFs
# =========================
st.title("Chat with your PDFs")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_texts = []
    st.session_state.docs = []
    for uf in uploaded_files:
        pages = pdf_to_text_chunks(uf.read())
        for p in pages:
            st.session_state.docs.append({"text": p})
            all_texts.append(p)
    if all_texts:
        st.session_state.embeds = embed_texts(all_texts)
        st.success(f"Loaded {len(uploaded_files)} PDF(s) with {len(all_texts)} chunks.")
else:
    st.session_state.docs = []
    st.session_state.embeds = None

# =========================
# Input Row
# =========================
query = st.text_input("Ask about any uploaded PDF...", key="query_input")

st.markdown('<div class="ask-row">', unsafe_allow_html=True)
ask_clicked = st.button("Ask", key="askbutton")
if st.session_state.chat_history:
    if st.button("Clear chat", key="clearchat", help="Clear all messages"):
        st.session_state.chat_history = []
        st.experimental_rerun()
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# Handle Ask
# =========================
if ask_clicked and query.strip():
    st.session_state.chat_history.insert(0, {"role": "user", "content": query})
    if st.session_state.embeds is None or len(st.session_state.docs) == 0:
        st.session_state.chat_history.insert(0, {"role": "bot", "content": "Upload PDFs first."})
    else:
        qvec = embed_texts([query])[0]
        idx, sims = cosine_topk(qvec, st.session_state.embeds, TOP_K)
        if len(idx) == 0:
            st.session_state.chat_history.insert(0, {"role": "bot", "content": "I don't know."})
        else:
            context_text = "\n\n".join(st.session_state.docs[i]["text"] for i in idx)
            prompt = f"Context:\n{context_text}\n\nQuestion: {query}\nAnswer:"
            answer = groq_answer(prompt)
            st.session_state.chat_history.insert(0, {"role": "bot", "content": answer})
    st.experimental_rerun()

# =========================
# Show Chat (newest first at top)
# =========================
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div style='display:flex;align-items:center;margin-bottom:4px;'>{USER_SVG}&nbsp;{chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='display:flex;align-items:center;margin-bottom:4px;'>{BOT_SVG}&nbsp;{chat['content']}</div>", unsafe_allow_html=True)
