import os
import hashlib
import tempfile

import streamlit as st
from dotenv import load_dotenv

from google import genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 RAG Document Assistant with Guardrails")
st.write(
    "Upload a PDF, index its content into a vector store, "
    "and ask questions powered by Gemini."
)

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY was not found. Please add it to your .env file.")
    st.stop()


# ============================================================
# EMBEDDINGS (loaded once, reused across reruns)
# ============================================================

class LocalHFEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()

    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()


@st.cache_resource
def get_embeddings():
    return LocalHFEmbeddings()


@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)


# ============================================================
# INDEXING (runs only when a NEW file is uploaded)
# ============================================================

def build_index(file_bytes: bytes, file_id: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        documents = PyPDFLoader(tmp_path).load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        chunks = splitter.split_documents(documents)

        # Unique collection per file so reruns/uploads never mix or duplicate
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            collection_name=f"doc_{file_id}",
        )

        # MMR reduces near-duplicate results from overlapping chunks
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20},
        )

        return retriever, len(documents), len(chunks)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ============================================================
# PDF UPLOAD
# ============================================================

uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.md5(file_bytes).hexdigest()

    # Only rebuild when the file actually changed
    if st.session_state.get("file_id") != file_id:
        with st.spinner("Processing PDF and creating vector store..."):
            retriever, n_pages, n_chunks = build_index(file_bytes, file_id)

        st.session_state["file_id"] = file_id
        st.session_state["retriever"] = retriever
        st.session_state["n_pages"] = n_pages
        st.session_state["n_chunks"] = n_chunks

    st.success(
        f"Indexed {st.session_state['n_pages']} pages into "
        f"{st.session_state['n_chunks']} text chunks."
    )

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    user_query = st.text_input("Ask a question about the uploaded document:")

    if user_query:
        with st.spinner("Retrieving relevant information and querying Gemini..."):
            relevant_docs = st.session_state["retriever"].invoke(user_query)

            context = "\n\n".join(doc.page_content for doc in relevant_docs)

            prompt = f"""
You are a precise document analysis assistant.

Your task is to answer the user's question using ONLY
the information contained in the provided document context.

Rules:

1. Do not use outside knowledge.
2. Do not make assumptions.
3. Do not invent information.
4. If the answer is not explicitly supported by the
   provided context, respond exactly with:

"I cannot answer this based on the provided document."

5. Give a clear and concise answer.
6. When possible, explain the answer using information
   from the retrieved context.

-------------------------
DOCUMENT CONTEXT
-------------------------

{context}

-------------------------
USER QUESTION
-------------------------

{user_query}
"""

            try:
                response = get_gemini_client().models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt,
                )

                st.subheader("Answer")
                st.write(response.text)

                with st.expander("Retrieved Context Chunks"):
                    for idx, doc in enumerate(relevant_docs):
                        st.markdown(f"### Chunk {idx + 1}")
                        st.write(doc.page_content)

            except Exception as e:
                st.error(f"Error querying Gemini API: {str(e)}")