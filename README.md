# Income Tax QA System using Retrieval-Augmented Generation (RAG) with Mistral 7B

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline that allows users to query legal documents—like *The Income Tax Act, 1961*—and receive natural language answers. The system combines the power of semantic retrieval and generative language models to answer user queries with high relevance and contextual awareness.

---

## 🚀 Features

- 💬 Question Answering on Legal PDFs  
- 🔍 Semantic Search using all-MiniLM-L6-v2 embeddings  
- 🧠 Context-aware generation using Mistral 7B (via Hugging Face)  
- 🧾 PDF ingestion and intelligent chunking for vectorization  
- ⚡ High-speed retrieval with ChromaDB Vector Store

---

## 🛠️ Tech Stack

- Python
- LangChain
- Hugging Face Transformers
- Sentence Transformers
- Chroma Vector Store
- PyTorch
- PyPDFLoader

---

## 📂 Project Workflow

1. Load and parse PDF using PyPDFLoader
2. Split text into manageable chunks using CharacterTextSplitter
3. Generate vector embeddings using all-MiniLM-L6-v2
4. Store vectors in ChromaDB for efficient retrieval
5. Accept user questions and retrieve top relevant chunks
6. Use Mistral 7B to generate natural language answers

---

## ⚙️ Setup Instructions

### 1. Clone Repository and Install Dependencies

```bash
pip install langchain langchain_community sentence-transformers chromadb torch torchvision torchaudio transformers accelerate pypdf
