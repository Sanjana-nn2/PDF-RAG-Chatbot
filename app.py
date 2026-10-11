
import streamlit as st
import requests
from io import BytesIO
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# -------------------------------
# Initialize chat history
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# -------------------------------
# Create embeddings using Ollama
# -------------------------------

class OllamaEmbeddings(Embeddings):

    def embed_documents(self, texts):
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text",
                "input": texts
            },
            timeout=120
        )

        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# -------------------------------
# Create cached vector store
# -------------------------------

@st.cache_resource(show_spinner=False)
def create_vector_store_v2(pdf_bytes):

    pdf = PdfReader(BytesIO(pdf_bytes))
    documents = []

    chunk_size = 500
    chunk_overlap = 50

    for page_number, page in enumerate(pdf.pages, start=1):

        page_text = page.extract_text()

        if not page_text or not page_text.strip():
            continue

        start = 0

        while start < len(page_text):
            end = start + chunk_size
            chunk = page_text[start:end]

            documents.append(
                Document(
                    page_content=chunk,
                    metadata={"page": page_number}
                )
            )

            if end >= len(page_text):
                break

            start = end - chunk_overlap

    if not documents:
        raise ValueError(
            "No readable text found in this PDF."
        )

    embeddings = OllamaEmbeddings()

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings
    )

    return vector_store


# -------------------------------
# PDF upload and question input
# -------------------------------

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

question = st.text_input(
    "Ask a question about your PDF:"
)


# -------------------------------
# Process the question
# -------------------------------

if st.button("Get Answer"):

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:

            # Prepare the PDF
            with st.spinner("Preparing your PDF..."):

                pdf_bytes = uploaded_file.getvalue()

                vector_store = create_vector_store_v2(
                    pdf_bytes
                )

                chunk_count = vector_store._collection.count()

            # Retrieve relevant PDF chunks
            with st.spinner("Finding relevant information..."):

                results = vector_store.similarity_search(
                    question,
                    k=min(5, chunk_count)
                )

                context = "\n\n".join(
                    f"[PDF Page {document.metadata.get('page', 'Unknown')}]\n"
                    f"{document.page_content}"
                    for document in results
                )

            # Generate the answer using Ollama
            with st.spinner("Generating your answer..."):

                prompt = f"""
You are a helpful PDF assistant.

Answer the user's question using only the information
provided in the context.

Rules:
1. Give a clear, direct answer in simple English.
2. Use only facts found in the provided context.
3. Do not invent facts or page numbers.
4. If the answer is not available in the context, say:
   "I could not find this information in your PDF."
5. If the context only partially answers the question,
   explain the limitation.

Context:
{context}

Question:
{question}

Answer:
"""

                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2:3b",
                        "prompt": prompt,
                        "stream": False
                    },
                    timeout=180
                )

                response.raise_for_status()

                answer = response.json()["response"].strip()

            # Add source page citations automatically
            source_pages = sorted({
                document.metadata.get("page")
                for document in results
                if document.metadata.get("page") is not None
            })

            # Check whether the answer indicates missing information
            missing_answer = (
                "I could not find this information in your PDF."
                in answer
            )

            if not missing_answer and source_pages:
                pages_text = ", ".join(
                    str(page) for page in source_pages
                )

                answer += f"\n\nSource pages: {pages_text}"

            # Save question and answer to chat history
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # Display the current question and answer
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                st.write(answer)

            # Display retrieved text and PDF page numbers
            with st.expander("View retrieved PDF text"):

                for i, document in enumerate(results, start=1):

                    page_number = document.metadata.get(
                        "page",
                        "Unknown"
                    )

                    st.markdown(
                        f"**Relevant section {i} — PDF Page {page_number}**"
                    )

                    st.write(document.page_content)

        except Exception as e:

            st.error(f"Something went wrong: {e}")

            st.info(
                "Check that Ollama is running and these models "
                "are installed: nomic-embed-text and llama3.2:3b."
            )
