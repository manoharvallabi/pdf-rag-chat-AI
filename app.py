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

DEFAULT_EMBED_DIM = 384
TOP_K_DEFAULT = 3
MAX_NEW_TOKENS_DEFAULT = 128
GROQ_MODEL = "llama-3.1-8b-instant"
SIM_THRESHOLD = 0.15

# =========================
# Text utilities
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
            buf = carry + [p]; cur = sum(len(x) + 1 for x in buf)
    if buf:
        chunks.append(" ".join(buf))
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

def build_prompt(context, question):
    return (
        "Use the provided context to answer as much as possible. "
        "If the question is not answered by the context, respond naturally as a helpful assistant.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )

# =========================
# Groq & Embeddings
# =========================
@st.cache_resource
def get_groq():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(embedder, texts):
    if isinstance(texts, str):
        texts = [texts]
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def cosine_topk(q, M, k):
    if M.size == 0:
        return np.array([], dtype=int), np.array([])
    sims = (M @ q.reshape(-1, 1)).ravel()
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def groq_generate(prompt, max_tokens=MAX_NEW_TOKENS_DEFAULT):
    client = get_groq()
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Be concise and helpful. Use context when given."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# =========================
# Streamlit App
# =========================
st.set_page_config(page_title="Chat with PDFs", page_icon="📘")
st.title("📘 Chat with Your PDFs")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "docs_all" not in st.session_state:
    st.session_state.docs_all = []
if "embeds" not in st.session_state:
    st.session_state.embeds = None

uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    embedder = get_embedder()
    current_names = {f.name for f in uploaded_files}
    st.session_state.docs_all = [d for d in st.session_state.docs_all if d["name"] in current_names]

    for file in uploaded_files:
        if any(doc["name"] == file.name for doc in st.session_state.docs_all):
            continue

        data = file.read()
        pages = pdf_to_pages(data)
        pairs = []
        for i, page_text in enumerate(pages, start=1):
            if not page_text:
                continue
            paras = split_paragraphs(page_text)
            chunks = merge_to_chunks(paras)
            for c in chunks:
                if c.strip():
                    pairs.append({"text": c, "page": i})

        texts = [p["text"] for p in pairs]
        if texts:
            with st.spinner(f"Indexing {file.name}..."):
                vecs = embed_texts(embedder, texts)
            st.session_state.docs_all.extend([{"name": file.name, "text": t} for t in texts])
            st.session_state.embeds = (
                vecs if st.session_state.embeds is None
                else np.vstack([st.session_state.embeds, vecs])
            )
    st.caption("Loaded files: " + ", ".join(sorted({d["name"] for d in st.session_state.docs_all})))

# =========================
# Query & Chat
# =========================
query = st.text_input("Ask a question", placeholder="Ask about any uploaded PDF…")

col1, col2 = st.columns([1, 5])
ask_clicked = col1.button("Ask")
clear_clicked = col2.button("Clear chat") if len(st.session_state.chat_history) > 0 else False

if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

if ask_clicked and query:
    msg = query.strip().lower()
    embedder = get_embedder()

    # Smart greetings & info
    if msg == "hi":
        answer = "Hi, happy to help — start your questions!"
    elif "hi" in msg and len(msg.split()) <= 3:
        answer = "Please continue with your question."
    elif "your name" in msg:
        answer = "I'm your document assistant, powered by Groq."
    elif "how many" in msg and ("document" in msg or "pdf" in msg):
        count = len({d["name"] for d in st.session_state.docs_all})
        answer = f"You currently have {count} PDF document{'s' if count != 1 else ''} uploaded."
    elif msg.startswith("summarize"):
        text_all = " ".join(d["text"] for d in st.session_state.docs_all)
        prompt = f"Summarize this document:\n\n{text_all[:12000]}"
        with st.spinner("Summarizing..."):
            answer = groq_generate(prompt, MAX_NEW_TOKENS_DEFAULT * 3)
    else:
        if st.session_state.embeds is None or st.session_state.embeds.size == 0:
            answer = "Please upload a PDF first."
        else:
            qvec = embed_texts(embedder, query)[0]
            idx, sims = cosine_topk(qvec, st.session_state.embeds, TOP_K_DEFAULT)
            if len(idx) == 0 or max(sims) < SIM_THRESHOLD:
                prompt = query
            else:
                context_text = "\n\n---\n\n".join(
                    [st.session_state.docs_all[i]["text"] for i in idx]
                )
                prompt = build_prompt(context_text, query)
            with st.spinner("Thinking..."):
                answer = groq_generate(prompt, MAX_NEW_TOKENS_DEFAULT)

    st.session_state.chat_history.insert(0, {"role": "assistant", "content": answer})
    st.session_state.chat_history.insert(0, {"role": "user", "content": query})
    st.rerun()

# =========================
# Chat Display (UI preserved)
# =========================
svg_user = """
<svg style='width:32px;height:32px;' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20'>
<g fill='none' stroke='currentColor' stroke-linecap='round'>
<circle cx='9.5' cy='5.5' r='3'/>
<path d='M15 16.5v-2c0-3.098-2.495-6-5.5-6c-3.006 0-5.5 2.902-5.5 6v2'/>
</g></svg>
"""

svg_bot = """
<svg style='width:32px;height:32px;' xmlns='http://www.w3.org/2000/svg' viewBox='0 0 2048 2048'>
<path fill='currentColor' d='M640 768h128v128H640V768zm512 0h128v128h-128V768zm469 640q35 0 66 13t54 37t37 55t14 66v469h-128v-469q0-18-12-30t-31-13H299q-18 0-30 12t-13 31v469H128v-469q0-35 13-66t37-54t54-37t67-14h341v-128h-85q-35 0-66-13t-55-37t-36-54t-14-67v-85H256V768h128v-85q0-35 13-66t37-54t54-37t67-14h341V303q-29-17-46-47t-18-64q0-27 10-50t27-40t41-28t50-10q27 0 50 10t40 27t28 41t10 50q0 34-17 64t-47 47v209h341q35 0 66 13t54 37t37 55t14 66v85h128v256h-128v85q0 35-13 66t-37 55t-55 36t-66 14h-85v128h341zM512 1109q0 18 12 30t31 13h810q18 0 30-12t13-31V683q0-18-12-30t-31-13H555q-18 0-30 12t-13 31v426zm256 299h384v-128H768v128z'/>
</svg>
"""

for msg in st.session_state.chat_history:
    icon = svg_user if msg["role"] == "user" else svg_bot
    st.markdown(
        f"""
        <div style='display:flex;align-items:flex-start;margin-bottom:12px;'>
          <div style='width:32px;height:32px;flex:0 0 32px;'>{icon}</div>
          <div style='margin-left:8px;flex:1;'>{msg['content']}</div>
        </div>
        """,
        unsafe_allow_html=True
    )
