import io, os, re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from sentence_transformers import SentenceTransformer
from openai import OpenAI


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_EMBED_MODEL = st.secrets.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

client = OpenAI(api_key=OPENAI_API_KEY)

TOP_K_DEFAULT = 3
SIM_THRESHOLD = 0.12     
MAX_TOKENS = 250         


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

def merge_to_chunks(paras, target=950, overlap=150):
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
            except:
                t = ""
            pages.append(clean_text(t))
    except:
        pages = []

    # If PDF is image-like → fallback to pdfminer
    if sum(len(p) for p in pages) < 300:
        try:
            full = pdfminer_extract_text(io.BytesIO(data)) or ""
            pages = [clean_text(x) for x in full.split("\f")]
        except:
            pass

    return pages


def embed_openai_batch(texts):
    res = client.embeddings.create(
        input=texts,
        model=OPENAI_EMBED_MODEL
    )
    return np.array([np.array(e.embedding) for e in res.data])

# fallback sentence-transformers
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def embed_text(texts):
    try:
        return embed_openai_batch(texts)
    except:
        embedder = get_embedder()
        return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def cosine_topk(q, M, k):
    if M.size == 0:
        return np.array([], dtype=int), np.array([])
    sims = (M @ q.reshape(-1, 1)).ravel()
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]


def openai_chat(prompt):
    res = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=MAX_TOKENS
    )
    return res.choices[0].message.content.strip()

def build_prompt(ctx, q):
    return f"""
You are a strict PDF question-answering assistant.

Answer ONLY using the PDF context below.
If the answer is not explicitly in the context, respond EXACTLY with:
"Not found in the document."

Context:
{ctx}

Question:
{q}

Answer:
"""


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
    current_names = {f.name for f in uploaded_files}
    st.session_state.docs_all = [d for d in st.session_state.docs_all if d["name"] in current_names]

    for f in uploaded_files:
        if any(d["name"] == f.name for d in st.session_state.docs_all):
            continue

        data = f.read()
        pages = pdf_to_pages(data)

        pairs = []
        for i, pg in enumerate(pages, start=1):
            if not pg.strip():
                continue
            paras = split_paragraphs(pg)
            chunks = merge_to_chunks(paras)
            for c in chunks:
                if c.strip():
                    pairs.append({"text": c, "page": i})

        texts = [p["text"] for p in pairs]

        if texts:
            with st.spinner(f"Indexing {f.name}…"):
                vecs = embed_text(texts)

            st.session_state.docs_all.extend(
                [{"name": f.name, "text": t} for t in texts]
            )

            st.session_state.embeds = (
                vecs if st.session_state.embeds is None
                else np.vstack([st.session_state.embeds, vecs])
            )

    st.caption("Loaded files: " + ", ".join(sorted({d["name"] for d in st.session_state.docs_all})))


query = st.text_input("Ask a question", placeholder="Ask about any uploaded PDF…")

col1, col2 = st.columns([1, 5])
ask_clicked = col1.button("Ask")
clear_clicked = col2.button("Clear chat") if st.session_state.chat_history else False

if clear_clicked:
    st.session_state.chat_history = []
    st.rerun()

if ask_clicked and query:
    q = query.strip()

    # generate embedding
    qvec = embed_text([q])[0]

    if st.session_state.embeds is None or st.session_state.embeds.size == 0:
        answer = "Please upload a PDF first."
    else:
        idx, sims = cosine_topk(qvec, st.session_state.embeds, TOP_K_DEFAULT)

        if len(idx) == 0 or max(sims) < SIM_THRESHOLD:
            # no context match → avoid hallucinations
            answer = "Not found in the document."
        else:
            ctx = "\n\n---\n\n".join(
                st.session_state.docs_all[i]["text"] for i in idx
            )
            prompt = build_prompt(ctx, q)
            with st.spinner("Thinking…"):
                answer = openai_chat(prompt)

    st.session_state.chat_history.insert(0, {"role": "assistant", "content": answer})
    st.session_state.chat_history.insert(0, {"role": "user", "content": query})
    st.rerun()


svg_user = """<svg style='width:32px;height:32px;' … ></svg>"""
svg_bot = """<svg style='width:32px;height:32px;' … ></svg>"""

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
