import io, os, time, re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from huggingface_hub import InferenceClient
from groq import Groq

# =========================
# Config
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN", "")            # optional but recommended
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")    # required for answers

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384
TOP_K_DEFAULT = 3
MAX_NEW_TOKENS_DEFAULT = 128
LOW_CONFIDENCE = 0.08

GROQ_MODELS = [
    "llama-3.1-8b-instant",  # updated working model
    "gemma2-9b-it",
]

# =========================
# UI + CSS
# =========================
st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")

st.markdown("""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css">
<style>
/* Make clear chat button look like text */
div.stButton > button[kind="secondary"] {
    background: none !important;
    border: none !important;
    color: #0073e6 !important;
    text-decoration: underline !important;
    font-size: 14px !important;
    padding: 0 !important;
    cursor: pointer !important;
}
</style>
""", unsafe_allow_html=True)

st.title("Chat with your PDFs")

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

def split_paragraphs(text: str):
    paras = [clean_text(p) for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]

def merge_to_chunks(paras, target=900, overlap=120):
    chunks, buf, cur = [], [], 0
    for p in paras:
        if cur + len(p) + 1 <= target or not buf:
            buf.append(p); cur += len(p) + 1
        else:
            chunks.append(" ".join(buf))
            carry, acc = [], 0
            for para in reversed(buf):
                carry.insert(0, para); acc += len(para) + 1
                if acc >= overlap: break
            buf = carry + [p]
            cur = sum(len(x) + 1 for x in buf)
    if buf: chunks.append(" ".join(buf))
    return chunks

def pdf_to_pages(data: bytes):
    pages = []
    try:
        reader = PdfReader(io.BytesIO(data))
        for p in reader.pages:
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            pages.append(clean_text(t))
    except Exception:
        pages = []

    if sum(len(p or "") for p in pages) >= 500:
        return pages

    try:
        text_all = pdfminer_extract_text(io.BytesIO(data)) or ""
        pages_pm = [clean_text(x) for x in text_all.split("\f")]
        if sum(len(p or "") for p in pages_pm) > sum(len(p or "") for p in pages):
            return pages_pm
    except Exception:
        pass
    return pages

def build_prompt(context: str, question: str) -> str:
    return (
        "Answer strictly using the provided context. "
        "If the answer is not in the context, reply with \"I don't know\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )

# =========================
# Embeddings
# =========================
@st.cache_resource
def get_hf_client(model_id, token):
    return InferenceClient(model=model_id, token=token)

class HFEmbedder:
    def __init__(self, model_id, token, default_dim=DEFAULT_EMBED_DIM):
        self.client = get_hf_client(model_id, token)
        self.default_dim = default_dim
        self._dim = None

    def _pool(self, out):
        if isinstance(out, list) and out:
            if isinstance(out[0], list):
                arr = np.asarray(out, dtype=np.float32)
                return arr.mean(axis=0)
            return np.asarray(out, dtype=np.float32)
        return np.asarray([], dtype=np.float32)

    def encode_one(self, text, retries=3, backoff=0.6):
        last_err = None
        for i in range(retries):
            try:
                out = self.client.feature_extraction(text)
                vec = self._pool(out)
                if vec.size:
                    if self._dim is None:
                        self._dim = int(vec.shape[-1])
                    return vec.astype(np.float32)
            except Exception as e:
                last_err = e
            time.sleep(backoff * (i + 1))
        dim = self._dim or self.default_dim
        return np.zeros((dim,), dtype=np.float32)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = [self.encode_one(t) for t in texts]
        dim = self._dim or self.default_dim
        vecs = [v if v.size else np.zeros((dim,), dtype=np.float32) for v in vecs]
        return np.vstack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)

def normalize_rows(M):
    if M.size == 0:
        return M
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / n

def cosine_topk(q, M, k):
    if M.size == 0 or q.size == 0:
        return np.array([], dtype=int), np.array([])
    qn = q / (np.linalg.norm(q) + 1e-9)
    Mn = normalize_rows(M)
    sims = (Mn @ qn.reshape(-1, 1)).ravel()
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

# =========================
# Groq
# =========================
@st.cache_resource
def get_groq():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in Secrets.")
    return Groq(api_key=GROQ_API_KEY)

def groq_generate_sync(prompt: str, max_tokens: int):
    client = get_groq()
    for model in GROQ_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are concise and only use the given context."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip(), model
        except Exception:
            continue
    return "I don't know", "Groq"

# =========================
# Session State Init
# =========================
if "docs" not in st.session_state:
    st.session_state.docs = []
if "embeds" not in st.session_state:
    st.session_state.embeds = None
if "last_files" not in st.session_state:
    st.session_state.last_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# Upload
# =========================
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    names = [f.name for f in uploaded_files]
    if names != st.session_state.last_files:
        # reset
        st.session_state.docs, st.session_state.embeds = [], None
        st.session_state.last_files = names
        all_chunks = []
        for f in uploaded_files:
            pages = pdf_to_pages(f.getvalue())
            for i, page_text in enumerate(pages, start=1):
                paras = split_paragraphs(page_text)
                chunks = merge_to_chunks(paras, 900, 120)
                for c in chunks:
                    all_chunks.append({"text": c, "page": i, "file": f.name})
        texts = [p["text"] for p in all_chunks]
        if texts:
            embedder = HFEmbedder(EMBED_MODEL, HF_TOKEN)
            vecs = embedder.encode(texts)
            st.session_state.docs = all_chunks
            st.session_state.embeds = vecs
st.write("Loaded files:", ", ".join(st.session_state.last_files) if st.session_state.last_files else "None")

# =========================
# Chat Interface
# =========================
query = st.text_input("Ask about any uploaded PDF...", key="query_box")

if st.button("Ask"):
    if st.session_state.embeds is None or len(st.session_state.docs) == 0:
        st.error("Upload PDFs first.")
        st.stop()
    embedder = HFEmbedder(EMBED_MODEL, HF_TOKEN)
    qvec = embedder.encode(query)
    idx, sims = cosine_topk(qvec[0], st.session_state.embeds, TOP_K_DEFAULT)
    context_text = "\n\n".join(st.session_state.docs[i]["text"] for i in idx)
    prompt = build_prompt(context_text, query)
    text, model = groq_generate_sync(prompt, MAX_NEW_TOKENS_DEFAULT)
    # store in history (latest first)
    st.session_state.chat_history.insert(0, {"role": "user", "content": query})
    st.session_state.chat_history.insert(0, {"role": "assistant", "content": text})

# =========================
# Clear Chat Button
# =========================
if len(st.session_state.chat_history) > 0:
    if st.button("Clear chat", key="clear_chat", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

# =========================
# Render Chat History (Newest First)
# =========================
if st.session_state.chat_history:
    for msg in st.session_state.chat_history:  # already newest first
        if msg["role"] == "user":
            icon_html = "<i class='bi bi-person'></i>"
        else:
            icon_html = "<i class='bi bi-robot'></i>"
        st.markdown(
            f"<div style='display:flex;align-items:flex-start;gap:8px;margin-bottom:8px;'>"
            f"<span style='font-size:20px;'>{icon_html}</span>"
            f"<div>{msg['content']}</div></div>",
            unsafe_allow_html=True
        )
