# src/retriever.py
from __future__ import annotations
import os, hashlib
import chromadb
from chromadb.utils import embedding_functions
from docling.document_converter import DocumentConverter
from docling_core.types.doc import ImageRefMode
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class PDFIndexerRetriever:
    def __init__(self, collection_name: str = "pdfs_rag"):
        # Chroma persistente
        self.client = chromadb.PersistentClient(path="data/chroma_store")
        
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # self.collection = self.client.get_or_create_collection(
        #     name=collection_name,
        #     embedding_function=ef,  # <- chave: a collection agora sabe embedar
        # )
        # self.client.delete_collection(name="pdfs_rag")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=ef,
        )

        self.converter = DocumentConverter()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def ensure_ready(self):
        # checagem leve; falha cedo se banco estiver vazio
        if self.collection.count() == 0:
            return True
        return False

    def load_pdfs(self, pdf_paths):
        all_docs = []
        for path in pdf_paths:
            dldoc = self.converter.convert(path).document
            basename = os.path.basename(path)

            # número total de páginas
            n_pages = dldoc.num_pages()

            for p in range(1, n_pages + 1):
                md_text = dldoc.export_to_markdown(
                    page_no=p,  # <-- exporta só a página p
                    image_mode=ImageRefMode.PLACEHOLDER,
                    image_placeholder=''
                )
                all_docs.append(
                    Document(
                        page_content=md_text,
                        metadata={"source": basename, "page": p}
                    )
                )
        return all_docs




    def _stable_id(self, text: str, meta: dict) -> str:
        # ID estável para evitar duplicar ao reindexar
        base = f"{meta.get('source','pdf')}|{meta.get('page','')}"
        h = hashlib.sha1((base + "|" + text.strip()).encode("utf-8")).hexdigest()
        return f"doc_{h}"

    def chunk_and_index(self, docs):
        chunks = self.splitter.split_documents(docs)
        texts = [d.page_content for d in chunks]
        metadatas = []
        for idx, d in enumerate(chunks):
            m = dict(d.metadata or {})
            m.setdefault("source", "pdf")
            m.setdefault("page", None)
            m["chunk"] = idx
            metadatas.append(m)

        ids = [self._stable_id(t, m) for t, m in zip(texts, metadatas)]
        self.collection.upsert(documents=texts, metadatas=metadatas, ids=ids)


    def build_index_from_folder(self, folder_path: str) -> None:
        pdfs = [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(".pdf")
        ]
        if not pdfs:
            print("Nenhum PDF encontrado em", folder_path)
            return
        print("Processando PDFs:", pdfs)
        docs = self.load_pdfs(pdfs)
        self.chunk_and_index(docs)

    def retrieve(self, query: str, k: int = 5):
        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas", "distances"],  # <- sem "ids"
        )

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        ids   = res.get("ids", [[]])[0]   # <- pegue aqui, não no include

        hits = []
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            sim = 1.0 - float(dist)  # se métrica = distância de cosseno
            hits.append({
                "id": _id,
                "text": doc,
                "meta": meta,
                "distance": dist,
                "similarity": sim
            })
        # ordenar por similaridade (maior é melhor)
        hits.sort(key=lambda h: h["similarity"], reverse=True)
        return hits

if __name__ == "__main__":
    retriever = PDFIndexerRetriever()

    retriever.build_index_from_folder("data/pdfs")

    query = "O que é autenticação?"
    results = retriever.retrieve(query, k=3)

    print("\n🔎 Resultados da busca:")
    for r in results:
        src  = r['meta'].get('source')
        page = r['meta'].get('page')
        sim  = r['similarity']
        snippet = r['text'][:300].replace("\n", " ") + ("..." if len(r['text']) > 300 else "")
        print(f"- Fonte: {src} (pág {page}) | Similaridade: {sim:.3f}")
        print(snippet, "\n")
