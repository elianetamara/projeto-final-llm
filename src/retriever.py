import os
import chromadb
from chromadb.config import Settings
from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


class PDFIndexerRetriever:
    def __init__(self, collection_name="pix_sources"):
        self.client = chromadb.Client(
            Settings(chroma_db_impl="duckdb+parquet",
                     persist_directory="data/chroma_store")
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name)
        self.embed = HuggingFaceEmbeddings(
            model_name="sentence-transformers/gte-small")
        self.converter = DocumentConverter()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)

    def load_pdfs(self, pdf_paths):
        all_docs = []
        for path in pdf_paths:
            doc = self.converter.convert(path).document
            md_text = doc.export_to_markdown()
            langchain_doc = Document(
                page_content=md_text,
                metadata={"source": os.path.basename(path)}
            )
            all_docs.append(langchain_doc)
        return all_docs

    def chunk_and_index(self, docs):
        docs_chunks = self.splitter.split_documents(docs)

        texts = [d.page_content for d in docs_chunks]
        metadatas = [
            {"source": d.metadata.get("source", "pdf"),
             "page": d.metadata.get("page", None)}
            for d in docs_chunks
        ]
        ids = [f"doc_{i}" for i in range(len(texts))]

        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)
        self.client.persist()
        print(f"Textos indexados no Chroma: {len(texts)}")

    def build_index_from_folder(self, folder_path):
        pdfs = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".pdf")
        ]
        if not pdfs:
            print("Nenhum PDF encontrado em ", folder_path)
            return
        print("Processando PDFs:", pdfs)
        docs = self.load_pdfs(pdfs)
        self.chunk_and_index(docs)

    def retrieve(self, query, k=5):
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        hits = []
        for doc, meta, dist in zip(res['documents'][0], res['metadatas'][0], res['distances'][0]):
            hits.append({"text": doc, "meta": meta, "score": dist})
        return hits


if __name__ == "__main__":
    retriever = PDFIndexerRetriever()

    retriever.build_index_from_folder("data/pdfs")

    query = "O que é bloqueio cautelar do Pix?"
    results = retriever.retrieve(query, k=3)

    print("\n🔎 Resultados da busca:")
    for r in results:
        print(
            f"- Fonte: {r['meta'].get('source')} (pág {r['meta'].get('page')}) | Score: {r['score']:.4f}")
        print(r["text"][:300], "...\n")
