# -*- coding: utf-8 -*-
"""Income_Tax.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NomGNh9ne2nQx8loKjHUEwP3lKkWf_E4
"""

pip install pypdf

pip install langchain_community

from langchain_community.document_loaders import PyPDFLoader

!pip install langchain

loader = PyPDFLoader(r"/content/drive/MyDrive/Income Tax.pdf")
pages = loader.load_and_split()

pages

type(pages)

len(pages)

!pip install --upgrade langchain_community
!pip install --upgrade langchain

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

pip install sentence_transformers

from langchain.text_splitter import SpacyTextSplitter

text_splitter = SpacyTextSplitter(chunk_size=500)

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
separator = "\n",
chunk_size = 500,
chunk_overlap = 100,
length_function = len,
is_separator_regex = False,
)

texts = text_splitter.split_documents(documents=pages)

print(texts[0])

print(len(texts))

type(texts)

pip install sentence-transformers langchain chromadb

pip install -U langchain-huggingface

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

!pip uninstall -y tensorflow transformers
!pip install tensorflow-cpu==2.12.0
!pip install transformers==4.33.2
!pip uninstall -y tensorflow
!pip install torch torchvision torchaudio
!pip install transformers
!pip uninstall -y tensorflow
!pip install tensorflow-cpu

pip install sentence-transformers

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(texts, embedding_model)

print(f"Created a vectorstore with {len(texts)} documents.")

pip install langchain chromadb sentence-transformers torch torchvision torchaudio transformers accelerate

from huggingface_hub import login
login(token="YOUR_HUGGINGFACE_TOKEN")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

def retrieve_context(question, retriever, top_k=5):
    """
    Retrieve the top-k most relevant documents from the vector store.
    """
    docs = retriever.get_relevant_documents(question)
    context = "\n".join([doc.page_content for doc in docs[:top_k]])
    return context

def generate_answer_with_mistral(question, context, model, tokenizer):
    """
    Use Mistral 7B to generate an answer based on the retrieved context.
    """
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

    # Generating response
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
    """
    RAG pipeline: retrieve context and generate an answer.
    """
    # Step 1: Retrieving relevant context
    context = retrieve_context(question, retriever, top_k)

    if not context:
        return "Sorry, I couldn't find relevant information in the documents."

    # Step 2: Generating answer using Mistral 7B
    answer = generate_answer_with_mistral(question, context, model, tokenizer)
    return answer

from langchain.vectorstores import Chroma

# Loading the Chroma vector store
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

question = "What are the general provisions mentioned in the Act?"

# Get the response
answer = rag_pipeline(question, retriever, model, tokenizer)
print(f"Question: {question}")
print(f"Answer: {answer}")

