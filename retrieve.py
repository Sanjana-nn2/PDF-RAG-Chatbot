
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


embeddings = OllamaEmbeddings()

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

question = input("Ask a question about your PDF: ")

results = vector_store.similarity_search(question, k=3)

print("\nRelevant information from your PDF:\n")

for i, document in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(document.page_content)
