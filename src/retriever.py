import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings

CHROMA_DIR = "data/chroma_db"
EMBED_MODEL = "sentence-transformers/gte-small"

class Retriever:
    def __init__(self):
        self.client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=CHROMA_DIR))
        self.collection = self.client.get_collection("pix_sources")
        self.embed = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    def retrieve(self, query, k=5):
        # dense retrieval via chroma query
        res = self.collection.query(query_texts=[query], n_results=k, include=["documents","metadatas","distances"])
        hits = []
        for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            hits.append({"text":doc, "meta":meta, "score":dist})
        return hits
