import io
import os
import time
import numpy as np
import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from huggingface_hub import InferenceClient
from groq import Groq
import re

# =========================
# Config
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384
TOP_K_DEFAULT = 3
MAX_NEW_TOKENS_DEFAULT = 128

GROQ_MODEL = "llama-3.1-8b-instant"

# =========================
# SVG Icons
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
# Utils
# =========================
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def pdf_to_pages(data: bytes) -> list[str]:
    pages = []
    try:
        reader = PdfReader(io.BytesIO(data))
        for p in reader.pages:
            t = p.extract_text() or ""
            pages.append(clean_text(t))
    except Exception:
        pages = []
    if sum(len(p or "") for p in pages) < 500:
        try:
            text_all = pdfminer_extract_text(io.BytesIO(data)) or ""
            pages = [clean_text(x) for x in text_all.split("\f")]
        except Exception:
            pass
    return pages

@st.cache_resource
def get_embedder():
    return InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)

@st.cache_resource
def get_groq():
    return Groq(api_key=GROQ_API_KEY)

def cosine_topk(q, M, k):
    if M.size == 0 or q.size == 0:
        return [], []
    qn = q / (np.linalg.norm(q) + 1e-9)
    Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    sims = (Mn @ qn.reshape(-1, 1)).ravel()
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

def build_prompt(context, question):
    return f"""Answer strictly using the provided context. 
If the answer is not in the context, reply "I don't know".

Context:
{context}

Question: {question}
Answer:"""

# =========================
# App
# =========================
st.set_page_config(page_title="PDF Chat", page_icon="📄")

if "docs" not in st.session_state:
    st.session_state.docs = []
if "embeds" not in st.session_state:
    st.session_state.embeds = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("Chat with your PDFs")

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.docs = []
    texts = []
    for file in uploaded_files:
        data = file.read()
        pages = pdf_to_pages(data)
        for p in pages:
            if p.strip():
                st.session_state.docs.append({"text": p})
                texts.append(p)
    if texts:
        embedder = get_embedder()
        vecs = np.vstack([np.mean(embedder.feature_extraction(t), axis=0) for t in texts])
        st.session_state.embeds = vecs
        st.success(f"Loaded {len(uploaded_files)} PDF(s) with {len(texts)} chunks.")

query = st.text_input("Ask about any uploaded PDF...")

col1, col2 = st.columns([1,5])
with col1:
    ask_clicked = st.button("Ask")
with col2:
    if st.button("Clear chat"):
        st.session_state.chat_history = []

# display newest first
for chat in reversed(st.session_state.chat_history):
    if chat["role"] == "user":
        st.markdown(f"<div style='display:flex;align-items:center;'>{USER_SVG}&nbsp;{chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div style='display:flex;align-items:center;'>{BOT_SVG}&nbsp;{chat['content']}</div>", unsafe_allow_html=True)

if ask_clicked and query.strip():
    st.session_state.chat_history.append({"role": "user", "content": query})
    if st.session_state.embeds is None:
        st.session_state.chat_history.append({"role": "bot", "content": "Upload PDFs first."})
    else:
        embedder = get_embedder()
        qvec = np.mean(embedder.feature_extraction(query), axis=0)
        idx, sims = cosine_topk(qvec, st.session_state.embeds, TOP_K_DEFAULT)
        if not idx.any():
            context_text = ""
        else:
            context_text = "\n\n".join(st.session_state.docs[i]["text"] for i in idx)
        prompt = build_prompt(context_text, query)
        client = get_groq()
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": "Answer concisely from context only"},
                      {"role": "user", "content": prompt}],
            max_tokens=MAX_NEW_TOKENS_DEFAULT,
        )
        answer = resp.choices[0].message.content.strip()
        st.session_state.chat_history.append({"role": "bot", "content": answer})
