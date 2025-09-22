import io, os, re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from groq import Groq
from sentence_transformers import SentenceTransformer

# =========================
# Config
# =========================
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not set.")
    st.stop()

DEFAULT_EMBED_DIM = 384  # Local model outputs 384-dim vectors
TOP_K_DEFAULT = 3
MAX_NEW_TOKENS_DEFAULT = 128
GROQ_MODELS = ["llama-3.1-8b-instant"]

# =========================
# Text utils
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

def merge_to_chunks(paras: list[str], target: int = 900, overlap: int = 120) -> list[str]:
    chunks, buf = [], []
    cur = 0
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

def pdf_to_pages(data: bytes) -> list[str]:
    pages = []
    try:
        reader = PdfReader(io.BytesIO(data))
        for p in reader.pages:
            try: t = p.extract_text() or ""
            except: t = ""
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
    except: pass
    return pages

def build_prompt(context: str, question: str) -> str:
    return (
        "Use the provided context to answer as much as possible. "
        "If the question is not answered by the context, respond naturally as a helpful assistant. "
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

# =========================
# Groq client
# =========================
@st.cache_resource
def get_groq():
    return Groq(api_key=GROQ_API_KEY)

# =========================
# Local Embeddings (SentenceTransformer)
# =========================
@st.cache_resource
def load_local_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")  # fast and free

class LocalEmbedder:
    def __init__(self):
        self.model = load_local_embedder()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs

def encode_in_batches(embedder, texts, batch_size=50):
    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        vecs = embedder.encode(batch)
        all_vecs.append(vecs)
    return np.vstack(all_vecs)

def cosine_topk(q: np.ndarray, M: np.ndarray, k: int):
    if M.size == 0 or q.size == 0: return np.array([], dtype=int), np.array([])
    sims = (M @ q.reshape(-1, 1)).ravel()  # already normalized
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k - 1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

# =========================
# Groq generation
# =========================
def groq_generate(prompt: str, max_tokens: int):
    client = get_groq()
    for model in GROQ_MODELS:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content":
                 "You are a helpful, friendly assistant. Use the provided context when it is relevant, otherwise answer normally."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip(), model

# =========================
# App UI
# =========================
st.set_page_config(page_title="Chat with your PDFs", page_icon="🔎")
st.title("Chat with your PDFs")

if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "docs_all" not in st.session_state: st.session_state.docs_all = []
if "embeds" not in st.session_state: st.session_state.embeds = None

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

# Sync docs with uploader
if not uploaded_files:
    st.session_state.docs_all = []
    st.session_state.embeds = None
else:
    current_names = {f.name for f in uploaded_files}
    st.session_state.docs_all = [d for d in st.session_state.docs_all if d["name"] in current_names]
    embedder = LocalEmbedder()
    for file in uploaded_files:
        if any(doc["name"] == file.name for doc in st.session_state.docs_all):
            continue
        data = file.read()
        pages = pdf_to_pages(data)
        pairs = []
        for i, page_text in enumerate(pages, start=1):
            if not page_text: continue
            paras = split_paragraphs(page_text)
            chunks = merge_to_chunks(paras)
            for c in chunks:
                if c.strip(): pairs.append({"text": c, "page": i})
        texts = [p["text"] for p in pairs]
        if texts:
            with st.spinner(f"Indexing {file.name} ({len(texts)} chunks)…"):
                vecs = encode_in_batches(embedder, texts, batch_size=50)
            st.session_state.docs_all.extend([{"name": file.name, "text": t} for t in texts])
            if st.session_state.embeds is None:
                st.session_state.embeds = vecs
            else:
                st.session_state.embeds = np.vstack([st.session_state.embeds, vecs])

if st.session_state.docs_all:
    st.caption("Loaded files: " + ", ".join(sorted({d["name"] for d in st.session_state.docs_all})))

query = st.text_input("Ask a question", placeholder="Ask about any uploaded PDF…")
col1, col2 = st.columns([1,5])
ask_clicked = col1.button("Ask")
if len(st.session_state.chat_history) > 0:
    clear_clicked = col2.button("Clear chat", help="Clear chat", key="clear_chat", use_container_width=False)
else:
    clear_clicked = False

if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

# =========================
# Question handling
# =========================
if ask_clicked and query:
    msg = query.strip().lower()

    # quick responses
    if msg == "hi":
        answer = "Hi, happy to help — start your questions!"
        st.session_state.chat_history.insert(0, {"role": "assistant", "content": answer})
        st.session_state.chat_history.insert(0, {"role": "user", "content": query})
        st.rerun()

    if "your name" in msg:
        answer = "I'm a chatbot powered by Groq."
        st.session_state.chat_history.insert(0, {"role": "assistant", "content": answer})
        st.session_state.chat_history.insert(0, {"role": "user", "content": query})
        st.rerun()

    if "how many" in msg and ("document" in msg or "pdf" in msg):
        count = len({d["name"] for d in st.session_state.docs_all})
        answer = f"You currently have {count} PDF document{'s' if count != 1 else ''} uploaded."
        st.session_state.chat_history.insert(0, {"role": "assistant", "content": answer})
        st.session_state.chat_history.insert(0, {"role": "user", "content": query})
        st.rerun()

    if st.session_state.embeds is None or st.session_state.embeds.size == 0:
        st.error("Upload a PDF first.")
        st.stop()

    embedder = LocalEmbedder()

    # ---- SUMMARIZATION FALLBACK ----
    if msg.startswith("summarize"):
        full_text = " ".join(d["text"] for d in st.session_state.docs_all)
        full_text = full_text[:10000]  # avoid overload
        prompt = f"Summarize the following document:\n\n{full_text}"
        with st.spinner("Generating summary…"):
            answer, used_model = groq_generate(prompt, MAX_NEW_TOKENS_DEFAULT * 4)

    else:
        # ---- RAG MODE / NORMAL CHAT ----
        qvec = embedder.encode(query)
        idx, sims = cosine_topk(qvec[0], st.session_state.embeds, TOP_K_DEFAULT)

        SIM_THRESHOLD = 0.15
        if len(idx) == 0 or max(sims) < SIM_THRESHOLD:
            prompt = query
            with st.spinner("Thinking…"):
                answer, used_model = groq_generate(prompt, MAX_NEW_TOKENS_DEFAULT)
        else:
            context_text = "\n\n---\n\n".join([st.session_state.docs_all[i]["text"] for i in idx])
            prompt = build_prompt(context_text, query)
            with st.spinner("Thinking…"):
                answer, used_model = groq_generate(prompt, MAX_NEW_TOKENS_DEFAULT)

    st.session_state.chat_history.insert(0, {"role": "assistant", "content": answer})
    st.session_state.chat_history.insert(0, {"role": "user", "content": query})
    st.rerun()


# =========================
# Chat display — SVG UI (small icons)
# =========================
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        st.markdown(
            f"<div class='chat-row' style='display:flex;align-items:flex-start;'>"
            f"<svg class='icon' width='24' height='24' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20'><g fill='none' stroke='currentColor' stroke-linecap='round'><circle cx='9.5' cy='5.5' r='3'/><path d='M15 16.5v-2c0-3.098-2.495-6-5.5-6c-3.006 0-5.5 2.902-5.5 6v2'/></g></svg>"
            f"<span style='margin-left:8px;'>{msg['content']}</span></div>",
            unsafe_allow_html=True)
    else:
        st.markdown(
            f"<div class='chat-row' style='display:flex;align-items:flex-start;'>"
            f"<svg class='icon' width='24' height='24' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 2048 2048'><path fill='currentColor' d='M640 768h128v128H640V768zm512 0h128v128h-128V768zm469 640q35 0 66 13t54 37t37 55t14 66v469h-128v-469q0-18-12-30t-31-13H299q-18 0-30 12t-13 31v469H128v-469q0-35 13-66t37-54t54-37t67-14h341v-128h-85q-35 0-66-13t-55-37t-36-54t-14-67v-85H256V768h128v-85q0-35 13-66t37-54t54-37t67-14h341V303q-29-17-46-47t-18-64q0-27 10-50t27-40t41-28t50-10q27 0 50 10t40 27t28 41t10 50q0 34-17 64t-47 47v209h341q35 0 66 13t54 37t37 55t14 66v85h128v256h-128v85q0 35-13 66t-37 55t-55 36t-66 14h-85v128h341zM512 1109q0 18 12 30t31 13h810q18 0 30-12t13-31V683q0-18-12-30t-31-13H555q-18 0-30 12t-13 31v426zm256 299h384v-128H768v128z'/></svg>"
            f"<span style='margin-left:8px;'>{msg['content']}</span></div>",
            unsafe_allow_html=True)
