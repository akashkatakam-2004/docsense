# 📄 DocSense — AI Document Q&A

> Upload any PDF and chat with it instantly — powered by RAG, Groq LLM, and FAISS vector search.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green?logo=fastapi)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange)
![Groq](https://img.shields.io/badge/Groq-LLM-purple)
![FAISS](https://img.shields.io/badge/FAISS-VectorStore-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker)
![HuggingFace](https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow)

---

## 🚀 Live Demo
👉 [Try it on Hugging Face Spaces](https://huggingface.co/spaces/Akashkatakam/DocSense)

---

## ✨ Features

- 📤 **PDF Upload** — Upload any PDF up to 10MB
- 🧠 **RAG Pipeline** — Retrieval-Augmented Generation for accurate answers
- ⚡ **Streaming Responses** — Real-time token-by-token answer streaming
- 🔍 **FAISS Vector Search** — Semantic search over document chunks
- 💾 **Session Persistence** — FAISS indexes saved to disk, restored on restart
- 🤖 **Multiple LLM Models** — Switch between LLaMA, Mixtral, Gemma
- 🔐 **Basic Auth** — Protected endpoints via HTTP Basic Authentication
- 🧹 **Auto Cleanup** — Expired sessions removed every 30 minutes

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
PyPDF Loader → Text Splitter → FAISS Vector Store
                                      │
User Question ──────────────► Similarity Search (k=4)
                                      │
                                      ▼
                              Groq LLM (LLaMA 3.3)
                                      │
                                      ▼
                              Streamed Answer + Sources
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| RAG | LangChain + ConversationalRetrievalChain |
| LLM | Groq (LLaMA 3.3 70B, Mixtral, Gemma) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | FAISS (with disk persistence) |
| PDF Parsing | PyPDF |
| Deployment | Docker + HuggingFace Spaces |

---

## ⚙️ Local Setup

### 1. Clone the repo
```bash
git clone https://github.com/akashkatakam-2004/docsense.git
cd docsense
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and fill in your values
```

> 🔑 Get your free Groq API key at [console.groq.com](https://console.groq.com)

### 5. Run the app
```bash
python app.py
```

Open your browser at `http://localhost:7860`

---

## 🐳 Docker Setup

```bash
docker build -t docsense .
docker run -p 7860:7860 \
  -e GROQ_API_KEY=your_key \
  -e APP_USERNAME=admin \
  -e APP_PASSWORD=yourpassword \
  docsense
```

---

## 📂 Project Structure

```
docsense/
├── app.py           # FastAPI app (no auth — for HuggingFace)
├── main.py          # FastAPI app (with Basic Auth — for self-hosting)
├── rag.py           # RAG pipeline: PDF processing, FAISS, Q&A, streaming
├── requirements.txt # Python dependencies
├── Dockerfile       # Container setup
├── .env.example     # Environment variable template
├── static/          # Frontend static files
├── templates/       # HTML templates
└── README.md        # You are here
```

---

## 🔐 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key | ✅ Yes |
| `APP_USERNAME` | Basic auth username (main.py only) | Optional |
| `APP_PASSWORD` | Basic auth password (main.py only) | Optional |

---

## 🤖 Supported Models

| Model | Speed | Quality |
|-------|-------|---------|
| `llama-3.3-70b-versatile` | Medium | ⭐⭐⭐⭐⭐ |
| `llama-3.1-8b-instant` | Fast | ⭐⭐⭐ |
| `mixtral-8x7b-32768` | Medium | ⭐⭐⭐⭐ |
| `gemma2-9b-it` | Fast | ⭐⭐⭐ |

---

## 👨‍💻 Built by

**Akash Katakam** — [Hugging Face](https://huggingface.co/Akashkatakam)

---

## 📄 License

MIT License — feel free to use, modify, and share!
