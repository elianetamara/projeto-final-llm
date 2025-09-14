from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import chromadb
from chromadb.config import Settings
import os, json

# Configs
DATA_DIR = "data/sources"
CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "sentence-transformers/gte-small"  # ou bge-small

def load_pdfs(pdf_paths):
    docs = []
    for p in pdf_paths:
        loader = PyPDFLoader(p)
        docs.extend(loader.load())
    return docs

def load_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    return loader.load()

def chunk_and_index(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs_chunks = splitter.split_documents(docs)

    # Embeddings (HuggingFace)
    embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # Chroma client
    client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
    collection = client.get_or_create_collection(name="pix_sources", metadata={"source":"pix_pool"})

    texts = [d.page_content for d in docs_chunks]
    metadatas = [{"source": d.metadata.get("source") or d.metadata.get("source_url") or "pdf", "page": d.metadata.get("page",None)} for d in docs_chunks]
    collection.add(documents=texts, metadatas=metadatas, ids=[f"doc_{i}" for i in range(len(texts))])
    client.persist()
    print(f"Indexed {len(texts)} chunks into Chroma.")

if __name__ == "__main__":
    # Discover PDFs in data/sources
    pdfs = [os.path.join(DATA_DIR,f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".pdf")]
    print("Found pdfs:", pdfs)
    docs = load_pdfs(pdfs)
    chunk_and_index(docs)
