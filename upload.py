from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from bs4 import SoupStrainer

import os


# Prep documents to be uploaded to the vector database (Pinecone)
loader = DirectoryLoader('../', glob="**/*.pdf", loader_cls=PyPDFLoader)
raw_docs = loader.load()

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
documents = text_splitter.split_documents(raw_docs)
print(f"Going to add {len(documents)} to Pinecone")

# Choose the embedding model and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
PineconeVectorStore.from_documents(documents=documents, embedding=embeddings, index_name="web-dev-textbook")
print("Loading to vectorstore done")