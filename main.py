
import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3.2:3b",
        "prompt": "What is RAG in artificial intelligence? Explain in two simple sentences.",
        "stream": False
    }
)

result = response.json()
print(result["response"])
