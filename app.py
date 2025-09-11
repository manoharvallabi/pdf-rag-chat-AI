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
st.set_page_config(page_title="Chat with your PDFs", page_icon="🔎")
st.title("Chat with your PDFs")

HF_TOKEN = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBED_DIM = 384
TOP_K_DEFAULT = 3
MAX_NEW_TOKENS_DEFAULT = 128
LOW_CONFIDENCE = 0.08
GROQ_MODELS = ["llama-3.1-8b-instant"]  # stable

# =========================
# Styling
# =========================
st.markdown("""
<style>
.clear-button {
    background: none !important;
    border: none !important;
    padding: 0 !important;
    font-size: 14px !important;
    cursor: pointer;
    color: inherit !important;
    margin-left: 10px;
}
.chat-icon {
    flex-shrink: 0;
    width: 22px;
    height: 22px;
    margin-right:6px;
    display:inline-block;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Functions (same as before)
# =========================
# ... keep your clean_text, split_paragraphs, merge_to_chunks, pdf_to_pages,
# build_prompt, get_hf_client, get_groq, get_embedder, HFEmbedder,
# normalize_rows, cosine_topk, groq_generate exactly as before ...

# =========================
# Session State
# =========================
if "chat_history" not in st.session_state: st.session_state.chat_history = []
if "docs_all" not in st.session_state: st.session_state.docs_all = []
if "embeds" not in st.session_state: st.session_state.embeds = None

# =========================
# File uploader logic (same)
# =========================
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.session_state.docs_all = []
    st.session_state.embeds = None
else:
    current_names = {f.name for f in uploaded_files}
    st.session_state.docs_all = [d for d in st.session_state.docs_all if d["name"] in current_names]

    for file in uploaded_files:
        if any(doc["name"] == file.name for doc in st.session_state.docs_all):
            continue
        data = file.read()
        # extract & embed as before...
        pages = pdf_to_pages(data)
        pairs = []
        for i, page_text in enumerate(pages, start=1):
            if not page_text: continue
            paras = split_paragraphs(page_text)
            chunks = merge_to_chunks(paras)
            for c in chunks:
                if c.strip(): pairs.append({"text": c, "page": i})
        texts = [p["text"] for p in pairs]
        embedder = get_embedder(EMBED_MODEL, HF_TOKEN)
        vecs = embedder.encode(texts)
        st.session_state.docs_all.extend([{"name": file.name, "text": t} for t in texts])
        if st.session_state.embeds is None:
            st.session_state.embeds = vecs
        else:
            st.session_state.embeds = np.vstack([st.session_state.embeds, vecs])

if st.session_state.docs_all:
    st.caption("Loaded files: " + ", ".join(sorted({d["name"] for d in st.session_state.docs_all})))

# =========================
# Ask + Clear Chat Inline
# =========================
query_col, clear_col = st.columns([4,1])
with query_col:
    query = st.text_input("Ask a question", placeholder="Ask about any uploaded PDF…")
with clear_col:
    if len(st.session_state.chat_history) > 0:
        if st.button("Clear chat", key="clear_chat", help="Clear chat", use_container_width=True):
            st.session_state.chat_history=[]
            st.experimental_rerun()

if st.button("Ask"):
    if query:
        msg = query.strip().lower()

        # special: hi
        if msg == "hi":
            answer = "Hi, happy to help, start your questions" if len(st.session_state.chat_history)==0 else "please continue"
            st.session_state.chat_history.append({"role":"user","content":query})
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.experimental_rerun()

        # special: name
        if "your name" in msg:
            answer = "I'm a chatbot."
            st.session_state.chat_history.append({"role":"user","content":query})
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.experimental_rerun()

        # special: how many docs
        if "how many" in msg and ("document" in msg or "pdf" in msg):
            count = len({d["name"] for d in st.session_state.docs_all})
            answer = f"You currently have {count} PDF document{'s' if count!=1 else ''} uploaded."
            st.session_state.chat_history.append({"role":"user","content":query})
            st.session_state.chat_history.append({"role":"assistant","content":answer})
            st.experimental_rerun()

        if st.session_state.embeds is None or st.session_state.embeds.size==0:
            st.error("Upload a PDF first.")
            st.stop()

        embedder = get_embedder(EMBED_MODEL, HF_TOKEN)
        qvec = embedder.encode(query)
        idx, sims = cosine_topk(qvec[0], st.session_state.embeds, TOP_K_DEFAULT)
        if len(idx)==0:
            answer = "I don't know."
        else:
            context_text = "\n\n---\n\n".join([st.session_state.docs_all[i]["text"] for i in idx])
            prompt = build_prompt(context_text, query)
            answer, used_model = groq_generate(prompt, MAX_NEW_TOKENS_DEFAULT)

        st.session_state.chat_history.append({"role":"user","content":query})
        st.session_state.chat_history.append({"role":"assistant","content":answer})
        st.experimental_rerun()

# =========================
# Chat display (fixed icon size)
# =========================
person_icon = '<svg class="chat-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor"><g fill="none" stroke="currentColor" stroke-linecap="round"><circle cx="9.5" cy="5.5" r="3"/><path d="M15 16.5v-2c0-3.098-2.495-6-5.5-6c-3.006 0-5.5 2.902-5.5 6v2"/></g></svg>'
robot_icon = '<svg class="chat-icon" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2048 2048" fill="currentColor"><path d="M640 768h128v128H640V768zm512 0h128v128h-128V768zm469 640q35 0 66 13t54 37t37 55t14 66v469h-128v-469q0-18-12-30t-31-13H299q-18 0-30 12t-13 31v469H128v-469q0-35 13-66t37-54t54-37t67-14h341v-128h-85q-35 0-66-13t-55-37t-36-54t-14-67v-85H256V768h128v-85q0-35 13-66t37-54t54-37t67-14h341V303q-29-17-46-47t-18-64q0-27 10-50t27-40t41-28t50-10q27 0 50 10t40 27t28 41t10 50q0 34-17 64t-47 47v209h341q35 0 66 13t54 37t37 55t14 66v85h128v256h-128v85q0 35-13 66t-37 55t-55 36t-66 14h-85v128h341zM512 1109q0 18 12 30t31 13h810q18 0 30-12t13-31V683q0-18-12-30t-31-13H555q-18 0-30 12t-13 31v426zm256 299h384v-128H768v128z"/></svg>'

for msg in reversed(st.session_state.chat_history):  # newest at top
    icon = person_icon if msg["role"]=="user" else robot_icon
    st.markdown(f"<div style='display:flex;align-items:flex-start;margin-bottom:10px;'>{icon}<div>{msg['content']}</div></div>", unsafe_allow_html=True)
