# PDF RAG Chat — Hugging Face (Free)

A zero-cost, deployable doc chatbot. Upload a PDF, ask questions, get answers grounded in the document.

- **Embeddings:** Hugging Face Inference API (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM:** Hugging Face Inference API (`Qwen/Qwen2-1.5B-Instruct` by default)
- **Vector store:** In-memory ChromaDB
- **UI:** Streamlit

## Quickstart

### 1) Create Hugging Face token (free)
- Make an account at https://huggingface.co
- Settings → Access Tokens → **New token** (read)

### 2) Local run (optional)
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export HF_TOKEN=your_token_here  # Windows PowerShell: $env:HF_TOKEN="your_token_here"
streamlit run app.py
```

### 3) Deploy on Streamlit Community Cloud (free)
1. Push this repo to GitHub.
2. Go to https://streamlit.io/cloud → **New app**.
3. Select your repo/branch. Entrypoint: `app.py`.
4. App → **Settings → Secrets** → add:
```
HF_TOKEN="your_token_here"
```
5. Deploy. You’ll get a public URL.

### 4) Usage
- Upload a PDF, click **Index document**.
- Ask questions in natural language.
- The model answers using only retrieved chunks from your PDF.

## Model switches
- Embeddings input box: default `sentence-transformers/all-MiniLM-L6-v2`
- LLM input box: default `Qwen/Qwen2-1.5B-Instruct`
- If you use a T5/FLAN/MT0/BART model name, the app auto-switches to the text-to-text API.

## Troubleshooting
- **Token missing** → Add `HF_TOKEN` to Streamlit Secrets or env.
- **Slow/Rate-limited** → Keep small models; reduce `MAX_NEW_TOKENS`.
- **Weak answers** → Increase `TOP_K` to 5–6, ask specific questions, ensure your PDF has extractable text.

**License:** MIT 
