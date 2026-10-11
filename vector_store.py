
import requests
from pypdf import PdfReader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# Create a custom embedding class using Ollama
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


# Read the PDF
pdf = PdfReader("documents/sample.pdf")

full_text = ""

for page in pdf.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"


# Split the text into chunks
chunk_size = 500
chunk_overlap = 50

chunks = []
start = 0

while start < len(full_text):
    end = start + chunk_size
    chunks.append(full_text[start:end])
    start = end - chunk_overlap


# Convert chunks into documents
documents = [
    Document(page_content=chunk)
    for chunk in chunks
]


# Create the Chroma vector database
embeddings = OllamaEmbeddings()

vector_store = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("PDF chunks stored in Chroma successfully!")
print("Total chunks stored:", len(documents))
