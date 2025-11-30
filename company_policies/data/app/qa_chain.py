from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import Chroma

from .data.app.config import CHROMA_DIR, EMBEDDING_MODEL_NAME, TOP_K

_embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

def _embed_fn(texts):
    return _embedder.encode(texts, convert_to_numpy=True).tolist()

def get_retriever():
    db = Chroma(
        embedding_function=_embed_fn,
        persist_directory=CHROMA_DIR,
    )
    return db

def answer_question(question: str):
    db = get_retriever()
    results = db.similarity_search(question, k=TOP_K)

    # Very simple “answer”: stitched context + metadata
    answer_parts = []
    sources = []
    for r in results:
        answer_parts.append(r.page_content)
        sources.append(str(r.metadata))

    answer_text = "\n\n---\n\n".join(answer_parts)
    return {
        "answer": answer_text,
        "sources": sources,
    }
