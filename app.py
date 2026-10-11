
import streamlit as st
from io import BytesIO
from pypdf import PdfReader
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 PDF RAG Chatbot")
st.write("Upload a PDF and ask questions about its content.")


# -------------------------------
# Chat history
# -------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# -------------------------------
# Extract and split PDF text
# -------------------------------

def extract_pdf_chunks(pdf_bytes):
    pdf = PdfReader(BytesIO(pdf_bytes))
    chunks = []

    chunk_size = 500
    chunk_overlap = 50

    for page_number, page in enumerate(pdf.pages, start=1):
        page_text = page.extract_text()

        if not page_text or not page_text.strip():
            continue

        start = 0

        while start < len(page_text):
            end = start + chunk_size
            chunk_text = page_text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            if end >= len(page_text):
                break

            start = end - chunk_overlap

    if not chunks:
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be a scanned PDF without selectable text."
        )

    return chunks


# -------------------------------
# Retrieve relevant chunks
# -------------------------------

def retrieve_chunks(chunks, question, top_k=5):
    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2)
    )

    document_vectors = vectorizer.fit_transform(texts)
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    ).flatten()

    ranked_indices = similarities.argsort()[::-1]

    results = []

    for index in ranked_indices:
        if similarities[index] <= 0:
            break

        results.append(chunks[index])

        if len(results) >= top_k:
            break

    return results


# -------------------------------
# Generate answer with Groq
# -------------------------------

def generate_answer(question, results):
    context = "\n\n".join(
        f"[PDF Page {item['page']}]\n{item['text']}"
        for item in results
    )

    client = Groq(
        api_key=st.secrets["GROQ_API_KEY"]
    )

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a PDF question-answering assistant. "
                    "Answer only using the supplied PDF context. "
                    "Use clear, simple English. Do not invent facts. "
                    "If the context does not contain the answer, "
                    "reply exactly: "
                    "I could not find this information in your PDF."
                )
            },
            {
                "role": "user",
                "content": (
                    f"PDF context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ],
        temperature=0.1
    )

    answer = response.choices[0].message.content

    if not answer:
        raise ValueError("The AI service returned an empty answer.")

    return answer.strip()


# -------------------------------
# PDF upload and question input
# -------------------------------

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)

question = st.text_input(
    "Ask a question about your PDF:",
    placeholder="e.g. Which CYP enzymes are important in drug metabolism?"
)


# -------------------------------
# Answer the question
# -------------------------------

if st.button("Get Answer", type="primary"):

    if uploaded_file is None:
        st.warning("Please upload a PDF first.")

    elif not question.strip():
        st.warning("Please enter a question.")

    else:
        try:
            with st.spinner("Reading your PDF..."):
                pdf_bytes = uploaded_file.getvalue()
                chunks = extract_pdf_chunks(pdf_bytes)

            with st.spinner("Searching the PDF..."):
                results = retrieve_chunks(
                    chunks,
                    question.strip(),
                    top_k=5
                )

            if not results:
                answer = (
                    "I could not find this information in your PDF."
                )

            else:
                with st.spinner("Generating your answer..."):
                    answer = generate_answer(question.strip(), results)

                missing_answer = (
                    "could not find this information in your PDF"
                    in answer.lower()
                )

                if not missing_answer:
                    source_pages = sorted({
                        item["page"] for item in results
                    })

                    answer += (
                        "\n\nSource pages: "
                        + ", ".join(map(str, source_pages))
                    )

            st.session_state.messages.append({
                "role": "user",
                "content": question.strip()
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("user"):
                st.write(question.strip())

            with st.chat_message("assistant"):
                st.write(answer)

            if results:
                with st.expander("View retrieved PDF text"):
                    for index, item in enumerate(results, start=1):
                        st.markdown(
                            f"**Relevant section {index} — "
                            f"PDF Page {item['page']}**"
                        )
                        st.write(item["text"])

        except Exception as error:
            st.error(f"Something went wrong: {error}")

            if "GROQ_API_KEY" in str(error):
                st.info(
                    "Check that GROQ_API_KEY is configured in "
                    "Streamlit Cloud Secrets."
                )
