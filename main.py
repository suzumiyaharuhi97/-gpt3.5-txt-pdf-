# 首先安装包：pip install openai langchain chromadb pypdf sentence_transformers
#

from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains.retrieval_qa.base import RetrievalQA
import os

os.environ["OPENAI_API_KEY"] = "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"

file_path = input("input pdf path: ")


loader = file_path.endswith(".pdf") and PyPDFLoader(file_path) or TextLoader(file_path, "utf-8")
print(loader)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = loader.load_and_split(splitter)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

db = Chroma.from_documents(chunks, embeddings)

llm = ChatOpenAI(temperature=0.7)

chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())

while True:
    question = input("\nQ: ")
    if not question:
        break
    print("A:", chain.run(question))
