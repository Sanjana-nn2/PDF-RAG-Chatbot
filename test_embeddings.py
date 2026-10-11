
import requests

response = requests.post(
    "http://localhost:11434/api/embed",
    json={
        "model": "nomic-embed-text",
        "input": "Python is a programming language."
    }
)

result = response.json()

embedding = result["embeddings"][0]

print("Embedding generated successfully!")
print("Number of values:", len(embedding))
print("First 5 values:", embedding[:5])
