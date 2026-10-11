# 📚 PDF RAG Chatbot

An AI-powered chatbot that allows users to upload PDF documents and ask questions about their content. The application retrieves relevant text from the uploaded PDF and uses a large language model (LLM) to generate answers with source page references.

## 🚀 Live Demo

**Try the application:** [PDF RAG Chatbot](https://pdf-rag-chatbot-vurifa2kkmhdsmtomenssn.streamlit.app/)

## ✨ Features

- 📄 Upload PDF documents directly through the web interface.
- 🔍 Extract text and divide it into smaller chunks.
- 🧠 Retrieve relevant text using TF-IDF and cosine similarity.
- 💬 Generate answers using the Groq API and a hosted language model.
- 📖 Display source page references for generated answers.
- 📝 View the retrieved PDF text used to answer questions.
- ☁️ Deploy and access the application online through Streamlit Community Cloud.

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Frontend / Web Framework:** Streamlit
- **PDF Processing:** pypdf
- **Text Retrieval:** Scikit-learn, TF-IDF, Cosine Similarity
- **LLM API:** Groq API
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git and GitHub

## ⚙️ How It Works

1. **Upload PDF:** The user uploads a PDF document.
2. **Text Extraction:** The application extracts readable text from the PDF.
3. **Chunking:** The extracted text is divided into smaller, overlapping chunks.
4. **Retrieval:** TF-IDF and cosine similarity identify text passages relevant to the user's question.
5. **Answer Generation:** The retrieved text is sent to a Groq-hosted language model to generate an answer.
6. **Source References:** The application displays the answer along with relevant PDF page references.

## 🖥️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Sanjana-nn2/PDF-RAG-Chatbot.git
cd PDF-RAG-Chatbot
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the Groq API key

Create a folder named `.streamlit` in the project directory. Inside it, create `secrets.toml` with the following content:

```toml
GROQ_API_KEY = "your_actual_groq_api_key"
```

Get your API key from the [Groq Console](https://console.groq.com/).

**Security:** Never commit your API key or `secrets.toml` file to GitHub.

### 5. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 📦 Dependencies

The main dependencies are:

- `streamlit`
- `pypdf`
- `groq`
- `scikit-learn`

They are listed in `requirements.txt`.

## 🔐 Security

- API credentials are stored in Streamlit secrets.
- The local secrets file should be excluded from version control.
- The application requires a valid Groq API key to generate answers.

## ⚠️ Limitations

- Retrieval uses TF-IDF, which primarily matches words and phrases rather than meaning.
- Scanned PDFs without extractable text are not supported by the current implementation.
- Answer quality depends on the extracted text, retrieved passages, and language model.
- Source pages indicate the pages of retrieved passages; they do not guarantee that every generated claim is correct.

## 🔮 Future Enhancements

- Integrate semantic embeddings for improved retrieval.
- Add a vector database such as ChromaDB.
- Support scanned PDFs using OCR.
- Improve source citations and answer verification.
- Add persistent document indexing and multi-document question answering.

## 👩‍💻 Author

**Sanjana N N**

- GitHub: [Sanjana-nn2](https://github.com/Sanjana-nn2)
- LinkedIn: [Sanjana N N](https://www.linkedin.com/in/sanjana-n-n-071413379/)

## 📄 License

This project is available for educational and portfolio purposes. Add a `LICENSE` file if you wish to distribute it under a specific open-source license.
