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
```

### 2. Authenticate with Hugging Face

```bash
from huggingface_hub import login
login(token="your_huggingface_token")
```

### 3. Load PDF and Preprocess

```bash
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/path/to/Income_Tax_Act.pdf")
pages = loader.load_and_split()
```

### 4. Split Text into Chunks

```bash
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

texts = text_splitter.split_documents(pages)
```

### 5. Create Vector Store

```bash
from langchain.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(texts, embedding_model)
```

## 🧠 Define the RAG Pipeline

```bash

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

def retrieve_context(question, retriever, top_k=5):
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs[:top_k]])
    return context

def generate_answer_with_mistral(question, context, model, tokenizer):
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

def rag_pipeline(question, retriever, model, tokenizer, top_k=5):
    context = retrieve_context(question, retriever, top_k)
    if not context:
        return "Sorry, I couldn't find relevant information in the documents."
    answer = generate_answer_with_mistral(question, context, model, tokenizer)
    return answer
```

## 🔎 Run the QA System

```bash
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

question = "What are the general provisions mentioned in the Act?"
answer = rag_pipeline(question, retriever, model, tokenizer)

print(f"Question: {question}")
print(f"Answer: {answer}")
```

## Sample Output
Question: What are the general provisions mentioned in the Act?
Answer: 1. The provisions relating to the levy and collection of tax.
2. The provisions relating to the assessment and collection of tax.
3. The provisions relating to the determination of taxable income.
4. The provisions relating to the procedure for appeals, revision, etc.
5. The provisions relating to the procedure for making rules.
6. The provisions relating to the procedure for making regulations.
7. The provisions relating to the grant of exemption.
8. The provisions relating to the levy and collection of interest, penalties and other sums.
9. The provisions relating to the grant of relief.
10. The provisions relating to the application of provisions of other laws.
11. The provisions relating to the application of provisions of this Act to a person not ordinarily resident in India.
12. The provisions relating to the application of provisions of this Act to a foreign company.
