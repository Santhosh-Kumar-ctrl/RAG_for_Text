import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

load_dotenv()

# load documents from the specified directory

def docsLoader(path):
    loader = PyPDFDirectoryLoader(path)
    documents = loader.load()
    return documents


# this function is used for chucking of the data

def chunker(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    return chunks


# this function is used for the embedding of the chunks and storing them in the chroma vector database

def embedding(chunks,presist_directory = "db/chroma_db"):
    persist_directory = presist_directory
    embedding_function = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = Chroma.from_documents(documents=chunks, embedding=embedding_function, persist_directory=persist_directory,collection_metadata={"hnsw:space": "cosine"})
    return vectordb

def main():
    docsPath = os.getenv("PDF_PATH")
    documents = docsLoader(docsPath)
    chunks = chunker(documents)
    for i, chunk in enumerate(chunks[:5]):
        print(f"-------Chunk {i+1}------")
        print(f"Source: {chunk.metadata['source']}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Content: {chunk.page_content}")
        print("-"*50)
    embedding(chunks)


if __name__ == "__main__":
    main()