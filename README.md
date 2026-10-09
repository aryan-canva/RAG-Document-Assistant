# 📚 RAG Document Assistant with Guardrails

A Streamlit app that lets you upload a PDF, index it into a local vector store, and ask questions answered by Google Gemini, strictly from the document's content.

If the answer isn't supported by the retrieved text, the assistant refuses instead of guessing.

## Features

- **PDF ingestion**: upload any text-based PDF and it is parsed page by page.
- **Chunking**: `RecursiveCharacterTextSplitter` (1000 characters, 200 overlap).
- **Local embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers`, so no embedding API cost and your documents' text is never sent out for embedding.
- **Vector search**: Chroma with MMR retrieval to avoid near-duplicate chunks.
- **Grounded answers**: Gemini is prompted to answer *only* from the retrieved context, otherwise reply exactly: `I cannot answer this based on the provided document.`
- **Transparency**: retrieved chunks are shown in an expander so you can verify where an answer came from.
- **Efficient reruns**: the index is built once per uploaded file (content-hashed) and cached in `st.session_state`.

## How it works


![RAG pipeline diagram](architecture.png)


1. **Index (once per file):** load → split → embed → store in a uniquely named Chroma collection.
2. **Query (every question):** retrieve the most relevant, diverse chunks → build a strict prompt → call Gemini → display the answer and the supporting chunks.

## Tech stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| PDF loading | PyPDF via LangChain |
| Chunking | LangChain text splitters |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | Chroma |
| LLM | Google Gemini (`google-genai`) |

## Setup

### 1. Clone and enter the project

```bash
cd rag-document-assistant
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
streamlit
python-dotenv
google-genai
langchain-community
langchain-core
langchain-text-splitters
chromadb
sentence-transformers
pypdf
```

### 4. Add your Gemini API key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_api_key_here
```

Get a key from [Google AI Studio](https://aistudio.google.com/). Make sure `.env` is in your `.gitignore`.

### 5. Run

```bash
streamlit run app.py --server.fileWatcherType none
```

The `--server.fileWatcherType none` flag is optional (see Troubleshooting).

## Usage

1. Upload a PDF.
2. Wait for the "Indexed N pages into M text chunks" message.
3. Ask a question about the content of the document.
4. Open **Retrieved Context Chunks** to see exactly what the answer was based on.

### Tips for better answers

- Ask about **content**, not structure. "What does the document say about errors and their propagation?" works much better than "solve assignment 2", because embeddings match meaning, not labels like numbers or headings.
- The assistant only knows what is in the PDF. If the PDF contains *questions* (e.g. an assignment sheet), it can tell you what is asked but will not solve it, since solving requires outside knowledge, which the guardrail forbids.

## Configuration

Key settings live in `app.py`:

| Setting | Default | Meaning |
|---|---|---|
| `chunk_size` | 1000 | Max characters per chunk |
| `chunk_overlap` | 200 | Characters shared between adjacent chunks |
| `k` | 4 | Chunks passed to the LLM |
| `fetch_k` | 20 | Candidate pool for MMR before picking `k` |
| `model` | set in `generate_content` | Gemini model name |

## Project structure

```
rag-document-assistant/
├── app.py            # Streamlit app (indexing + retrieval + generation)
├── requirements.txt
├── .env              # GEMINI_API_KEY (not committed)
└── README.md
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'torchvision'` in the terminal logs**
Harmless. Streamlit's file watcher inspects every `transformers` submodule, including vision ones this app doesn't use. Either run with `--server.fileWatcherType none` or `pip install torchvision`.

**Identical retrieved chunks / answers degrade as you ask more questions**
This came from re-indexing the PDF on every Streamlit rerun, which duplicated chunks in the same Chroma collection. The current version indexes once per file hash and uses a unique collection name per file.

**"I cannot answer this based on the provided document."**
Either the retrieved chunks don't contain the answer (rephrase using words from the document) or the question needs outside knowledge (by design, the guardrail refuses).

**Empty or missing text from some pages**
Scanned or image-only PDFs have no extractable text with PyPDF. You would need to add OCR (e.g. Tesseract) before indexing.

**`GEMINI_API_KEY was not found`**
Check that `.env` is in the directory you launch Streamlit from and the variable name matches exactly.

## Known limitations

- One PDF at a time; the index is in-memory and is lost when the session ends.
- Text-only: no OCR, tables and images are not handled specially.
- Prompt-based guardrails reduce hallucination but do not guarantee it, and a malicious PDF could contain prompt-injection text.
- No conversation memory; each question is answered independently.

## Roadmap ideas

- Persistent index (`persist_directory`) and multi-document support
- Page-number citations in answers
- Hybrid search (BM25 + embeddings) and re-ranking
- Optional "document + general knowledge" mode with clearly labelled sources
- Chat interface with follow-up question handling
- Evaluation set with retrieval and faithfulness metrics
