
from pypdf import PdfReader

# Step 1: Read the PDF
pdf = PdfReader("documents/sample.pdf")

# Step 2: Extract text from every page
full_text = ""

for page in pdf.pages:
    text = page.extract_text()
    if text:
        full_text += text + "\n"

# Step 3: Set chunk settings
chunk_size = 500
chunk_overlap = 50

# Step 4: Split text into chunks
chunks = []

start = 0

while start < len(full_text):
    end = start + chunk_size
    chunk = full_text[start:end]
    chunks.append(chunk)
    start = end - chunk_overlap

# Step 5: Display the results
print("Total characters:", len(full_text))
print("Total chunks:", len(chunks))

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
