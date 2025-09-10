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
GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL     = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
EMBED_BACKEND  = os.getenv("EMBEDDING_BACKEND", "LOCAL").upper()  # LOCAL | OPENAI | HF

HF_TOKEN       = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

EMBED_MODEL_HF     = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM  = 384
TOP_K_DEFAULT      = 3
MAX_NEW_TOKENS_DEF = 192
LOW_CONFIDENCE     = 0.02

# =========================
# Default UI
# =========================
st.set_page_config(page_title="Chat with your PDFs", page_icon="🔎")
st.title("Chat with your PDFs")

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

def build_prompt(context: str, question: str, strict: bool) -> str:
    if strict:
        rules = "Answer strictly using the provided context. If the answer is not in the context, reply with \"I don't know\"."
    else:
        rules = "Use the provided context primarily. If details are missing, answer helpfully."
    return f"{rules}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"

# =========================
# Clients (cached)
# =========================
@st.cache_resource
def groq_client():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in Secrets.")
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def hf_client(model_id: str, token: str | None):
    return InferenceClient(model=model_id, token=token)

@st.cache_resource
def local_st_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def openai_client():
    from openai import OpenAI
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=OPENAI_API_KEY)

# =========================
# Embeddings
# =========================
class Embedder:
    def __init__(self, backend: str):
        self.backend = backend
        self.dim = None
        if backend == "LOCAL":
            self.model = local_st_model()
            self.dim = 384
        elif backend == "OPENAI":
            self.cli = openai_client()
        elif backend == "HF":
            self.cli = hf_client(EMBED_MODEL_HF, HF_TOKEN)
        else:
            raise ValueError("EMBEDDING_BACKEND must be LOCAL, OPENAI, or HF")

    def encode(self, texts):
        if isinstance(texts, str): texts = [texts]
        if self.backend == "LOCAL":
            vecs = self.model.encode(texts, normalize_embeddings=False)
            arr = np.asarray(vecs, dtype=np.float32)
            self.dim = arr.shape[-1]
            return arr
        if self.backend == "OPENAI":
            out = self.cli.embeddings.create(model="text-embedding-3-small", input=texts)
            vecs = [np.array(d.embedding, dtype=np.float32) for d in out.data]
            arr = np.vstack(vecs); self.dim = arr.shape[-1]
            return arr
        if self.backend == "HF":
            def _pool(o):
                if isinstance(o, list) and o:
                    if isinstance(o[0], list):
                        a = np.asarray(o, dtype=np.float32); return a.mean(axis=0)
                    return np.asarray(o, dtype=np.float32)
                return np.asarray([], dtype=np.float32)
            vecs = []
            for t in texts:
                last = None
                for i in range(3):
                    try:
                        out = self.cli.feature_extraction(t)
                        v = _pool(out)
                        if v.size:
                            vecs.append(v.astype(np.float32)); break
                    except Exception as e:
                        last = e
                    time.sleep(0.5*(i+1))
                else:
                    dim = self.dim or DEFAULT_EMBED_DIM
                    vecs.append(np.zeros((dim,), dtype=np.float32))
                if not self.dim and vecs and vecs[-1].size:
                    self.dim = int(vecs[-1].shape[-1])
            dim = self.dim or DEFAULT_EMBED_DIM
            vecs = [v if v.size else np.zeros((dim,), dtype=np.float32) for v in vecs]
            return np.vstack(vecs)

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

def bm25_topk(query: str, docs: list[str], k: int):
    tokenized = [d.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())
    scores = np.asarray(scores, dtype=np.float32) if len(scores) else np.array([], dtype=np.float32)
    if scores.size == 0: return np.array([], dtype=int), np.array([])
    k = min(k, scores.size)
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx, scores[idx]

# =========================
# Groq generation
# =========================
def groq_stream(prompt: str, max_tokens: int):
    cli = groq_client()
    resp = cli.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are helpful, precise, and concise."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in resp:
        try:
            ch = chunk.choices[0]
            delta = getattr(ch, "delta", None)
            content = getattr(delta, "content", None) if delta else None
            if content: yield content
        except Exception:
            continue

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
# Sidebar
# =========================
with st.sidebar:
    st.subheader("Settings")
    strict_rag   = st.toggle("Strict RAG (only from doc)", value=False)
    fast_mode    = st.checkbox("Fast mode", value=True)
    long_answers = st.checkbox("Long answers", value=False)
    force_answer = st.checkbox("Answer at low confidence", value=True)
    show_debug   = st.checkbox("Show debug info", value=False)
    st.caption(f"Model: {GROQ_MODEL} • Embeddings: {EMBED_BACKEND}")

TOP_K = 2 if fast_mode else TOP_K_DEFAULT
MAX_NEW_TOKENS = 320 if long_answers else MAX_NEW_TOKENS_DEF
CONTEXT_CHAR_LIMIT = 3500 if fast_mode else 9000

# =========================
# Session
# =========================
if "docs_all" not in st.session_state:
    st.session_state.docs_all = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# Upload multiple PDFs
# =========================
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for file in uploaded_files:
        if any(doc["name"] == file.name for doc in st.session_state.docs_all):
            continue
        data = file.getvalue()
        pages = pdf_to_pages(data)
        pairs = []
        for i, page_text in enumerate(pages, start=1):
            paras = split_paragraphs(page_text)
            chunks = merge_to_chunks(paras, 900 if fast_mode else 1200, 120 if fast_mode else 180)
            for c in chunks:
                if c.strip():
                    pairs.append({"text": c, "page": i})
        texts = [p["text"] for p in pairs]
        vecs = Embedder(EMBED_BACKEND).encode(texts)
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
if all_embeds:
    embeds = np.vstack(all_embeds)
else:
    embeds = np.zeros((0, DEFAULT_EMBED_DIM), dtype=np.float32)

# =========================
# Display chat history
# =========================
for turn in st.session_state.chat_history:
    st.chat_message("user").write(turn["user"])
    st.chat_message("assistant").write(turn["answer"])

# =========================
# Ask form
# =========================
with st.form("qa", clear_on_submit=False):
    query = st.text_input("Ask a question", value="", placeholder="Ask about any uploaded PDF…")
    ready = embeds.size > 0
    submitted = st.form_submit_button("Ask", disabled=not ready)

if submitted:
    msg = (query or "").strip().lower()

    # Special question: how many documents
    if "how many" in msg and "document" in msg:
        count = len(st.session_state.docs_all)
        answer = f"You currently have {count} PDF document{'s' if count!=1 else ''} uploaded."
        st.session_state.chat_history.append({"user": query, "answer": answer})
        st.rerun()

    # Greeting rule
    if msg == "hi":
        answer = "Hi, happy to hep, start your questions" if len(st.session_state.chat_history)==0 else "please continue"
        st.session_state.chat_history.append({"user": query, "answer": answer})
        st.rerun()

    if not ready:
        st.error("Upload PDFs first.")
        st.stop()

    emb = Embedder(EMBED_BACKEND)
    qvec = emb.encode(query)
    dense_idx, dense_sims = cosine_topk(qvec[0], embeds, TOP_K)
    contexts = [all_pairs[i] for i in dense_idx] if len(dense_idx) else []
    best_sim = float(dense_sims.max()) if dense_sims.size else 0.0

    if best_sim < 0.02:
        all_texts = [d["text"] for d in all_pairs]
        bm_idx, bm_scores = bm25_topk(query, all_texts, TOP_K)
        if len(bm_idx):
            contexts = [all_pairs[i] for i in bm_idx]
            best_sim = max(best_sim, 0.05)

    if not contexts:
        answer = "Couldn't retrieve relevant context."
        st.session_state.chat_history.append({"user": query, "answer": answer})
        st.rerun()

    context_text = "\n\n---\n\n".join(x["text"] for x in contexts)
    if len(context_text) > CONTEXT_CHAR_LIMIT:
        context_text = context_text[:CONTEXT_CHAR_LIMIT]

    if strict_rag and not force_answer and (best_sim < LOW_CONFIDENCE or not context_text.strip()):
        answer = "I don't know"
        st.session_state.chat_history.append({"user": query, "answer": answer})
        st.rerun()

    prompt = build_prompt(context_text, query, strict=strict_rag)

    # generate
    try:
        chunks = []
        for tok in groq_stream(prompt, MAX_NEW_TOKENS):
            chunks.append(tok)
        answer = "".join(chunks)
    except Exception:
        answer = groq_generate_sync(prompt, MAX_NEW_TOKENS)

    st.session_state.chat_history.append({"user": query, "answer": answer})
    st.rerun()

# =========================
# Clear chat button only if history exists
# =========================
if len(st.session_state.chat_history) > 0:
    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()
