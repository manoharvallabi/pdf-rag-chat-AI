import io
import os
import time
import re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from huggingface_hub import InferenceClient
from groq import Groq

# =========================
# Config
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN", "")            # set in Streamlit Secrets
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")    # set in Streamlit Secrets

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # serverless embeddings
DEFAULT_EMBED_DIM = 384  # all-MiniLM-L6-v2 outputs 384-d vectors
TOP_K = 4
MAX_NEW_TOKENS = 256
LOW_CONFIDENCE = 0.08  # lower -> willing to answer more often

GROQ_MODELS = [
    "llama3-8b-8192",        # primary
    "llama-3.1-8b-instant",  # fallback (if enabled in your account)
    "gemma2-9b-it",          # fallback
]

# =========================
# Text utils
# =========================
def clean_text(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)   # join hyphenated line breaks
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)    # join intra-paragraph newlines
    s = re.sub(r"[ \t]{2,}", " ", s)         # collapse spaces
    return s.strip()

def split_paragraphs(text: str) -> list[str]:
    paras = [clean_text(p) for p in re.split(r"\n\s*\n", text)]
    return [p for p in paras if p]

def merge_to_chunks(paras: list[str], target: int = 1000, overlap: int = 150) -> list[str]:
    chunks, buf = [], []
    cur = 0
    for p in paras:
        if cur + len(p) + 1 <= target or not buf:
            buf.append(p); cur += len(p) + 1
        else:
            chunks.append(" ".join(buf))
            # overlap by reusing last ~overlap chars
            carry, acc = [], 0
            for para in reversed(buf):
                carry.insert(0, para); acc += len(para) + 1
                if acc >= overlap: break
            buf = carry + [p]
            cur = sum(len(x) + 1 for x in buf)
    if buf: chunks.append(" ".join(buf))
    return chunks

def pdf_to_pages(file) -> list[str]:
    """Try PyPDF; if little text, fall back to pdfminer."""
    data = file.read()

    # 1) PyPDF
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

    # 2) pdfminer (split on form-feed)
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
# Embeddings (HF serverless)
# =========================
class HFEmbedder:
    def __init__(self, model_id: str, token: str | None, default_dim: int = DEFAULT_EMBED_DIM):
        self.client = InferenceClient(model=model_id, token=token)
        self.default_dim = default_dim
        self._dim = None  # learn on first non-empty vector

    def _pool(self, out):
        # out may be: [float,...] or [[float,...], ...]
        if isinstance(out, list) and out:
            if isinstance(out[0], list):  # token-level -> mean pool
                arr = np.asarray(out, dtype=np.float32)
                vec = arr.mean(axis=0)
                return vec
            else:
                return np.asarray(out, dtype=np.float32)
        return np.asarray([], dtype=np.float32)

    def encode_one(self, text: str, retries: int = 4, backoff: float = 0.8) -> np.ndarray:
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
        # give a safe zero vector with known dim
        dim = self._dim or self.default_dim
        return np.zeros((dim,), dtype=np.float32)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = [self.encode_one(t) for t in texts]
        # If any vector ended up zero-length (shouldn't), fix shape.
        vecs = [v if v.size else np.zeros((self._dim or self.default_dim,), dtype=np.float32) for v in vecs]
        return np.vstack(vecs)

def normalize_rows(M: np.ndarray) -> np.ndarray:
    if M.size == 0:
        return M
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / n

def cosine_topk(q: np.ndarray, M: np.ndarray, k: int):
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
# Groq generation
# =========================
def groq_generate(prompt: str, max_tokens: int = MAX_NEW_TOKENS):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in Secrets.")
    client = Groq(api_key=GROQ_API_KEY)
    last_err = None
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
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Groq generation failed: {last_err}")

# =========================
# App
# =========================
st.set_page_config(page_title="PDF RAG (HF embeddings + Groq)", page_icon="📄")
st.title("📄 Chat with your PDF — HF embeddings + Groq generation")

if not HF_TOKEN:
    st.info("Tip: add HF_TOKEN in Secrets to avoid HF rate limits.")
if not GROQ_API_KEY:
    st.warning("Add GROQ_API_KEY in Secrets to enable answers.")

# session state
if "docs" not in st.session_state:
    st.session_state.docs = []         # list of {"text","page"}
if "embeds" not in st.session_state:
    st.session_state.embeds = None     # (N, d)
if "last_file" not in st.session_state:
    st.session_state.last_file = None

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
force_answer = st.checkbox("Answer even at low confidence", value=False)
show_debug = st.checkbox("Show debug info", value=False)

# auto-index after upload
def auto_index():
    with st.spinner("Reading and indexing…"):
        pages = pdf_to_pages(uploaded)
        pairs = []
        for i, page_text in enumerate(pages, start=1):
            if not page_text:
                continue
            paras = split_paragraphs(page_text)
            chunks = merge_to_chunks(paras, 1000, 150)
            for c in chunks:
                if c.strip():
                    pairs.append({"text": c, "page": i})

        texts = [p["text"] for p in pairs]
        if not texts:
            st.warning("No extractable text found. If your PDF is scanned, OCR it first.")
            st.session_state.docs, st.session_state.embeds = [], None
            return

        embedder = HFEmbedder(EMBED_MODEL, token=HF_TOKEN)
        vecs = embedder.encode(texts)  # (N, d)
        if vecs.size == 0 or vecs.ndim != 2:
            st.error("Embedding failed. Check HF_TOKEN or try again.")
            return

        st.session_state.docs = pairs
        st.session_state.embeds = vecs
        st.session_state.last_file = uploaded.name
        st.success(f"Indexed {len(texts)} chunks across {len(pages)} pages.")

if uploaded is not None:
    if st.session_state.last_file != uploaded.name or st.session_state.embeds is None:
        auto_index()

query = st.text_input("Ask a question about the PDF")
if st.button("Ask"):
    if st.session_state.embeds is None or len(st.session_state.docs) == 0:
        st.error("Upload a PDF first.")
        st.stop()

    # retrieve
    embedder = HFEmbedder(EMBED_MODEL, token=HF_TOKEN)
    qvec = embedder.encode(query)  # (1, d)
    if qvec.size == 0:
        st.error("Query embedding failed. Try again.")
        st.stop()

    idx, sims = cosine_topk(qvec[0], st.session_state.embeds, TOP_K)
    if len(idx) == 0:
        st.info("Couldn't retrieve relevant context. Try a different question.")
        st.stop()

    contexts = [st.session_state.docs[i] for i in idx]
    context_text = "\n\n---\n\n".join(x["text"] for x in contexts)
    best_sim = float(sims.max()) if sims.size else 0.0

    if show_debug:
        st.write({
            "top_cosine_scores": [float(s) for s in sims],
            "best_cosine": best_sim,
            "chunks": len(st.session_state.docs),
        })

    if not force_answer and (best_sim < LOW_CONFIDENCE or not context_text.strip()):
        st.markdown("### Answer")
        st.write("I don't know based on this document.")
        with st.expander("Show retrieved context"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i} (score: {float(sims[i-1]):.3f})**\n\n{c['text']}")
        st.stop()

    # generate
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY missing. Add it in Settings → Secrets.")
        st.stop()

    prompt = build_prompt(context_text, query)
    try:
        answer, used_model = groq_generate(prompt, MAX_NEW_TOKENS)
    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()

    # display
    st.markdown("### Answer")
    st.write(answer)
    st.caption(f"Groq model: {used_model}")

    with st.expander("Show retrieved context"):
        for i, c in enumerate(contexts, 1):
            st.markdown(f"**Chunk {i} (score: {float(sims[i-1]):.3f})**\n\n{c['text']}")
