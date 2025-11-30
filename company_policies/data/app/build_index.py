import os
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document

from .data.app.config import DATA_DIR, CHROMA_DIR, EMBEDDING_MODEL_NAME

def load_documents():
    docs = []
    # PDFs
    pdf_loader = DirectoryLoader(
        DATA_DIR,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs.extend(pdf_loader.load())

    # TXT
    txt_loader = DirectoryLoader(
        DATA_DIR,
        glob="*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs.extend(txt_loader.load())
    return docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)

def build_chroma_index():
    documents = load_documents()
    chunks = split_documents(documents)

    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
    def embed_fn(texts):
        return embedder.encode(texts, convert_to_numpy=True).tolist()

    # Wrap texts into Documents again with metadata
    chroma_docs = [
        Document(page_content=chunk.page_content, metadata=chunk.metadata)
        for chunk in chunks
    ]

    db = Chroma.from_documents(
        documents=chroma_docs,
        embedding_function=embed_fn,
        persist_directory=CHROMA_DIR,
    )
    db.persist()
    print(f"Indexed {len(chroma_docs)} chunks into {CHROMA_DIR}")

if __name__ == "__main__":
    os.makedirs(CHROMA_DIR, exist_ok=True)
    build_chroma_index()
