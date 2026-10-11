
from pypdf import PdfReader

pdf = PdfReader("documents/sample.pdf")

print("Number of pages:", len(pdf.pages))

for page in pdf.pages:
    text = page.extract_text()
    print(text)
