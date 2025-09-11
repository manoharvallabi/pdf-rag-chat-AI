import io, os, re
import numpy as np
import streamlit as st
from pypdf import PdfReader
from pdfminer.high_level import extract_text as pdfminer_extract_text
from huggingface_hub import InferenceClient
from groq import Groq

# ---------------- CONFIG ----------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384
TOP_K_DEFAULT = 3
MAX_NEW_TOKENS_DEFAULT = 128

GROQ_MODELS = ["llama-3.1-8b-instant"]

# ---------------- SVG ICONS ----------------
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

# ---------------- CSS ----------------
st.markdown("""
<style>
.chat-user, .chat-bot {
    padding: 0.4rem 0.5rem;
    margin: 0.2rem 0;
}
.chat-user svg, .chat-bot svg {
    vertical-align: middle;
    margin-right: 4px;
}
div[data-testid="stTextInput"] {
    flex: 1;
}
.clear-chat {
    background: none;
    border: none;
    color: #0073e6;
    text-decoration: underline;
    font-size: 14px;
    cursor: pointer;
    padding-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- UTILS ----------------
def clean_text(s: str) -> str:
    if not s: return ""
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
            try: t = p.extract_text() or ""
            except Exception: t = ""
            pages.append(clean_text(t))
    except Exception: pages = []
    if sum(len(p or "") for p in pages) >= 500:
        return pages
    try:
        text_all = pdfminer_extract_text(io.BytesIO(data)) or ""
        pages_pm = [clean_text(x) for x in text_all.split("\f")]
        if sum(len(p or "") for p in pages_pm) > sum(len(p or "") for p in pages):
            return pages_pm
    except Exception: pass
    return pages

def build_prompt(context: str, question: str) -> str:
    return (
        "Answer strictly using the provided context. "
        "If the answer is not in the context, reply with \"I don't know\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\nAnswer:"
    )

# ---------------- CACHE ----------------
@st.cache_resource
def get_hf_client(model_id, token):
    return InferenceClient(model=model_id, token=token)

@st.cache_resource
def get_groq():
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set")
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def get_embedder(model_id, token):
    return HFEmbedder(model_id, token)

# ---------------- EMBEDDINGS ----------------
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

    def encode_one(self, text: str):
        try:
            out = self.client.feature_extraction(text)
            vec = self._pool(out)
            if vec.size:
                if self._dim is None: self._dim = int(vec.shape[-1])
                return vec.astype(np.float32)
        except Exception: pass
        dim = self._dim or self.default_dim
        return np.zeros((dim,), dtype=np.float32)

    def encode(self, texts):
        if isinstance(texts, str): texts = [texts]
        vecs = [self.encode_one(t) for t in texts]
        dim = self._dim or self.default_dim
        vecs = [v if v.size else np.zeros((dim,), dtype=np.float32) for v in vecs]
        return np.vstack(vecs) if vecs else np.zeros((0, dim), dtype=np.float32)

def normalize_rows(M):
    if M.size == 0: return M
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    return M / n

def cosine_topk(q, M, k):
    if M.size == 0 or q.size == 0:
        return np.array([], dtype=int), np.array([])
    qn = q / (np.linalg.norm(q) + 1e-9)
    Mn = normalize_rows(M)
    sims = (Mn @ qn.reshape(-1,1)).ravel()
    k = min(k, len(sims))
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx, sims[idx]

# ---------------- GROQ ----------------
def groq_generate(prompt: str, max_tokens: int):
    client = get_groq()
    model = GROQ_MODELS[0]
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are concise and only use the given context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

# ---------------- APP ----------------
st.set_page_config(page_title="Chat with your PDFs", layout="centered")
st.title("Chat with your PDFs")

if "docs" not in st.session_state: st.session_state.docs = []
if "embeds" not in st.session_state: st.session_state.embeds = None
if "uploaded_bytes" not in st.session_state: st.session_state.uploaded_bytes = []
if "chat_history" not in st.session_state: st.session_state.chat_history = []

uploaded = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded:
    st.session_state.docs = []
    st.session_state.uploaded_bytes = []
    for file in uploaded:
        bytes_data = file.getvalue()
        pages = pdf_to_pages(bytes_data)
        for i, page_text in enumerate(pages, start=1):
            if not page_text: continue
            paras = split_paragraphs(page_text)
            chunks = merge_to_chunks(paras)
            for c in chunks:
                if c.strip():
                    st.session_state.docs.append({"text": c, "page": i})
        st.session_state.uploaded_bytes.append(bytes_data)

    texts = [p["text"] for p in st.session_state.docs]
    if texts:
        embedder = get_embedder(EMBED_MODEL, HF_TOKEN)
        vecs = embedder.encode(texts)
        st.session_state.embeds = vecs
        st.caption(f"Loaded files: {', '.join([f.name for f in uploaded])}")

# Show chat history oldest → newest
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='chat-user'>{USER_SVG} {chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='chat-bot'>{BOT_SVG} {chat['content']}</div>", unsafe_allow_html=True)

col1, col2 = st.columns([8,1])
with col1:
    query = st.text_input("Ask about any uploaded PDF...", key="query_input")
with col2:
    if st.session_state.chat_history:
        if st.button("Clear chat", key="clearchat", help="Clear chat",
                     use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.docs=[]
            st.session_state.embeds=None
            st.session_state.uploaded_bytes=[]

if st.button("Ask"):
    q = st.session_state.query_input
    st.session_state.chat_history.append({"role":"user","content":q})
    st.session_state.query_input = ""  # auto clear
    if st.session_state.embeds is None or len(st.session_state.docs)==0:
        st.session_state.chat_history.append({"role":"assistant","content":"Please upload PDFs first."})
    else:
        embedder = get_embedder(EMBED_MODEL, HF_TOKEN)
        qvec = embedder.encode(q)
        idx, sims = cosine_topk(qvec[0], st.session_state.embeds, TOP_K_DEFAULT)
        if len(idx)==0:
            answer="I don't know"
        else:
            context_text = "\n\n".join(st.session_state.docs[i]["text"] for i in idx)
            prompt = build_prompt(context_text, q)
            answer = groq_generate(prompt, MAX_NEW_TOKENS_DEFAULT)
        st.session_state.chat_history.append({"role":"assistant","content":answer})
