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
HF_TOKEN = os.environ.get("HF_TOKEN", "")            # set in Streamlit Secrets (optional but helps)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")    # set in Streamlit Secrets (required)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # HF serverless embeddings
DEFAULT_EMBED_DIM = 384
TOP_K_DEFAULT = 3          # fewer chunks -> smaller prompt -> faster
MAX_NEW_TOKENS_DEFAULT = 128
LOW_CONFIDENCE = 0.08

GROQ_MODELS = [
    "llama3-8b-8192",        # fast, default
    "gemma2-9b-it",          # solid fallback
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

def merge_to_chunks(paras: list[str], target: int = 900, overlap: int = 120) -> list[str]:
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
    data = file.read()

    # Try PyPDF
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

    # Fallback: pdfminer
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
# Cached clients/resources
# =========================
@st.cache_resource
def get_hf_client(model_id: str, token: str | None):
    return InferenceClient(model=model_id, token=token)

@st.cache_resource
def get_groq():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in Secrets.")
    return Groq(api_key=GROQ_API_KEY)

# =========================
# Embeddings (HF serverless)
# =========================
class HFEmbedder:
    def __init__(self, model_id: str, token: str | None, default_dim: int = DEFAULT_EMBED_DIM):
        self.client = get_hf_client(model_id, token)
        self.default_dim = default_dim
        self._dim = None

    def _pool(self, out):
        if isinstance(out, list) and out:
            if isinstance(out[0], list):
                arr = np.asarray(out, dtype=np.float32)
                return arr.mean(axis=0)
            else:
                return np.asarray(out, dtype=np.float32)
        return np.asarray([], dtype=np.float32)

    def encode_one(self, text: str, retries: int = 3, backoff: float = 0.6) -> np.ndarray:
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
# Groq generation (streaming)
# =========================
def groq_stream(prompt: str, model: str, max_tokens: int):
    client = get_groq()
    # Stream like OpenAI chunks
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are concise and only use the given context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
        stream=True,
    )
    for chunk in resp:
        try:
            delta = chunk.choices[0].delta
            if delta and getattr(delta, "content", None):
                yield delta.content
        except Exception:
            continue

def groq_generate_sync(prompt: str, max_tokens: int):
    client = get_groq()
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

# Speed controls
colA, colB, colC = st.columns(3)
with colA:
    fast_mode = st.checkbox("Fast mode", value=True, help="Smaller context, fewer tokens.")
with colB:
    long_answers = st.checkbox("Long answers", value=False, help="Increase max tokens if needed.")
with colC:
    force_answer = st.checkbox("Answer at low confidence", value=False)

TOP_K = 2 if fast_mode else TOP_K_DEFAULT
MAX_NEW_TOKENS = 256 if long_answers else MAX_NEW_TOKENS_DEFAULT
CONTEXT_CHAR_LIMIT = 3500 if fast_mode else 7000  # trim prompt length

if not GROQ_API_KEY:
    st.warning("Add GROQ_API_KEY in Secrets to enable answers.")
if not HF_TOKEN:
    st.info("Tip: add HF_TOKEN in Secrets to avoid HF embedding cold starts/rate limits.")

# session state
if "docs" not in st.session_state:
    st.session_state.docs = []         # list of {"text","page"}
if "embeds" not in st.session_state:
    st.session_state.embeds = None     # (N, d)
if "last_file" not in st.session_state:
    st.session_state.last_file = None

uploaded = st.file_uploader("Upload a PDF", type=["pdf"])
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
            chunks = merge_to_chunks(paras, 900 if fast_mode else 1100, 120 if fast_mode else 160)
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
    # Trim the prompt in fast mode to keep it snappy
    if len(context_text) > CONTEXT_CHAR_LIMIT:
        context_text = context_text[:CONTEXT_CHAR_LIMIT]

    best_sim = float(sims.max()) if sims.size else 0.0

    if show_debug:
        st.write({
            "top_cosine_scores": [float(s) for s in sims],
            "best_cosine": best_sim,
            "chunks": len(st.session_state.docs),
            "max_new_tokens": MAX_NEW_TOKENS,
            "top_k": TOP_K,
        })

    if not force_answer and (best_sim < LOW_CONFIDENCE or not context_text.strip()):
        st.markdown("### Answer")
        st.write("I don't know based on this document.")
        with st.expander("Show retrieved context"):
            for i, c in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i} (score: {float(sims[i-1]):.3f})**\n\n{c['text']}")
        st.stop()

    # generate (stream)
    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY missing. Add it in Settings → Secrets.")
        st.stop()

    prompt = build_prompt(context_text, query)

    st.markdown("### Answer")
    out_box = st.empty()
    collected = []

    # try streaming on the first model; if it errors fast, fall back to sync
    streamed = False
    for model in GROQ_MODELS:
        try:
            def _streamer():
                nonlocal streamed
                for tok in groq_stream(prompt, model, MAX_NEW_TOKENS):
                    streamed = True
                    collected.append(tok)
                    yield tok
            st.write_stream(_streamer())
            used_model = model
            break
        except Exception:
            continue

    if not streamed:
        # sync fallback
        try:
            text, used_model = groq_generate_sync(prompt, MAX_NEW_TOKENS)
            collected = [text]
            out_box.write(text)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            st.stop()

    st.caption(f"Groq model: {used_model}")

    with st.expander("Show retrieved context"):
        for i, c in enumerate(contexts, 1):
            st.markdown(f"**Chunk {i} (score: {float(sims[i-1]):.3f})**\n\n{c['text']}")
