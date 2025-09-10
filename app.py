import io
import os
import re
import time
import numpy as np
import streamlit as st

from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from rank_bm25 import BM25Okapi
from huggingface_hub import InferenceClient
from groq import Groq

# =========================
# Config
# =========================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = "llama-3.1-8b-instant"

HF_TOKEN     = os.getenv("HF_TOKEN", "")
EMBED_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384
TOP_K_DEFAULT = 3
MAX_NEW_TOKENS_DEF = 192
LOW_CONFIDENCE = 0.02

# =========================
# UI
# =========================
st.set_page_config(page_title="Chat with your PDFs", page_icon="📄")
st.title("📄 Chat with your PDFs")

# =========================
# Helpers
# =========================
def clean_text(s: str) -> str:
    if not s: return ""
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()

def split_paragraphs(text: str) -> list[str]:
    paras = [clean_text(p) for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]

def merge_to_chunks(paras: list[str], target: int = 1100, overlap: int = 160) -> list[str]:
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
            buf, cur = carry + [p], sum(len(x) + 1 for x in carry) + len(p) + 1
    if buf: chunks.append(" ".join(buf))
    return chunks

def pdf_to_pages(data: bytes) -> list[str]:
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
    if sum(len(p) for p in pages) >= 500:
        return pages
    try:
        all_text = pdfminer_extract_text(io.BytesIO(data)) or ""
        pages_pm = [clean_text(x) for x in all_text.split("\f")]
        if sum(len(p) for p in pages_pm) > sum(len(p) for p in pages):
            return pages_pm
    except Exception:
        pass
    return pages

def build_prompt(context: str, question: str) -> str:
    return (
        "Answer strictly using the provided context. "
        "If the answer is not in the context, reply with \"I don't know\".\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

def norm_rows(M: np.ndarray) -> np.ndarray:
    if M.size == 0: return M
    return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

def cosine_topk(q: np.ndarray, M: np.ndarray, k: int):
    if M.size == 0 or q.size == 0:
        return np.array([], dtype=int), np.array([])
    qn = q / (np.linalg.norm(q) + 1e-9)
    sims = (norm_rows(M) @ qn.reshape(-1, 1)).ravel()
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

# =========================
# Clients
# =========================
@st.cache_resource
def groq_client():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def hf_client():
    return InferenceClient(model=EMBED_MODEL, token=HF_TOKEN)

class Embedder:
    def __init__(self):
        self.cli = hf_client()
        self.dim = DEFAULT_EMBED_DIM
    def encode(self, texts):
        if isinstance(texts, str): texts = [texts]
        vecs = []
        for t in texts:
            out = self.cli.feature_extraction(t)
            arr = np.array(out, dtype=np.float32)
            if arr.ndim == 2: arr = arr.mean(axis=0)
            vecs.append(arr)
        return np.vstack(vecs)

def groq_generate_sync(prompt: str, max_tokens: int) -> str:
    cli = groq_client()
    resp = cli.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are helpful, precise, and concise."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# =========================
# Session state
# =========================
if "docs_all" not in st.session_state: st.session_state.docs_all = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# =========================
# Upload PDFs
# =========================
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Clear docs if nothing uploaded
if not uploaded_files:
    st.session_state.docs_all = []
else:
    # remove docs not present
    current_names = {f.name for f in uploaded_files}
    st.session_state.docs_all = [d for d in st.session_state.docs_all if d["name"] in current_names]

    # add new docs
    for file in uploaded_files:
        if any(doc["name"] == file.name for doc in st.session_state.docs_all):
            continue
        data = file.getvalue()
        pages = pdf_to_pages(data)
        pairs = []
        for i, page_text in enumerate(pages, start=1):
            paras = split_paragraphs(page_text)
            chunks = merge_to_chunks(paras)
            for c in chunks:
                if c.strip():
                    pairs.append({"text": c, "page": i})
        texts = [p["text"] for p in pairs]
        vecs = Embedder().encode(texts)
        st.session_state.docs_all.append({
            "name": file.name,
            "pages": len(pages),
            "pairs": pairs,
            "embeds": vecs
        })

if st.session_state.docs_all:
    st.caption("Loaded files: " + ", ".join(d["name"] for d in st.session_state.docs_all))

# Combine all docs
all_pairs = []
all_embeds = []
for doc in st.session_state.docs_all:
    for p, v in zip(doc["pairs"], doc["embeds"]):
        all_pairs.append(p)
        all_embeds.append(v)
embeds = np.vstack(all_embeds) if all_embeds else np.zeros((0, DEFAULT_EMBED_DIM), dtype=np.float32)

# =========================
# Chat history display
# =========================
for turn in st.session_state.chat_history:
    st.chat_message("user").write(turn["user"])
    st.chat_message("assistant").write(turn["answer"])

# =========================
# Ask a question form
# =========================
with st.form("qa", clear_on_submit=True):  # ✅ clears automatically
    query = st.text_input("Ask a question", placeholder="Ask about any uploaded PDF…")
    submitted = st.form_submit_button("Ask", disabled=embeds.size == 0)

if submitted:
    msg = (query or "").strip().lower()

    # greeting
    if msg == "hi":
        answer = "Hi, happy to hep, start your questions" if len(st.session_state.chat_history)==0 else "please continue"
        st.session_state.chat_history.append({"user": query, "answer": answer})
        st.rerun()

    # count docs
    if "how many" in msg and "document" in msg:
        count = len(st.session_state.docs_all)
        answer = f"You currently have {count} PDF document{'s' if count!=1 else ''} uploaded."
        st.session_state.chat_history.append({"user": query, "answer": answer})
        st.rerun()

    # retrieval
    emb = Embedder()
    qvec = emb.encode(query)
    dense_idx, dense_sims = cosine_topk(qvec[0], embeds, TOP_K_DEFAULT)
    contexts = [all_pairs[i] for i in dense_idx] if len(dense_idx) else []
    best_sim = float(dense_sims.max()) if dense_sims.size else 0.0

    if not contexts:
        answer = "Couldn't retrieve relevant context."
        st.session_state.chat_history.append({"user": query, "answer": answer})
        st.rerun()

    context_text = "\n\n---\n\n".join(x["text"] for x in contexts)
    if len(context_text) > 3500:
        context_text = context_text[:3500]

    prompt = build_prompt(context_text, query)

    # generate
    try:
        answer = groq_generate_sync(prompt, MAX_NEW_TOKENS_DEF)
    except Exception as e:
        answer = f"Generation failed: {e}"

    st.session_state.chat_history.append({"user": query, "answer": answer})
    st.rerun()

# =========================
# Clear chat button only if history exists
# =========================
if len(st.session_state.chat_history) > 0:
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
