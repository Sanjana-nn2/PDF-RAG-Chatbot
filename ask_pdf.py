
import requests
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings


class OllamaEmbeddings(Embeddings):

    def embed_documents(self, texts):
        response = requests.post(
            "http://localhost:11434/api/embed",
            json={
                "model": "nomic-embed-text",
                "input": texts
            }
        )
        response.raise_for_status()
        return response.json()["embeddings"]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# Connect to the existing Chroma database
embeddings = OllamaEmbeddings()

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Ask a question
question = input("Ask a question about your PDF: ")

# Retrieve the 3 most relevant chunks
results = vector_store.similarity_search(question, k=3)

# Combine retrieved chunks into context
context = "\n\n".join(
    document.page_content for document in results
)

# Ask Llama to answer using the PDF context
prompt = f"""
You are a helpful PDF assistant.

Answer the question using only the information in the provided context.
If the answer is not present in the context, say:
"I could not find this information in your PDF."

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
answer = response.json()["response"]

print("\nAnswer from your PDF:\n")
print(answer)
