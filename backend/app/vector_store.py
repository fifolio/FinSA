import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from app.embeddings import embed_texts

load_dotenv()

INDEX_NAME = "finsa"
DIMENSION = 384
_index = None

def get_index():
    global _index
    if _index is None:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        if not pc.has_index(INDEX_NAME):
            pc.create_index(
                name=INDEX_NAME,
                dimension=DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                            )
        _index = pc.Index(INDEX_NAME)
    return _index

def build_records(chunks: list[Document], vectors: list[list[float]]) -> list[dict]:
    records = []
    for chunk, vector in zip(chunks, vectors):
        meta = chunk.metadata
        records.append(
            {
                "id": f"{meta['company']}-{meta['year']}-{meta['chunk_id']}",
                "values": vector,
                "metadata": {
                    "company": meta["company"],
                    "year": meta["year"],
                    "source": meta["source"],
                    "page": meta.get("page", -1),
                    "text": chunk.page_content,
                },
            }
        )
    return records

def upsert_chunks(chunks: list[Document], batch_size: int = 100) -> int:
    vectors = embed_texts([c.page_content for c in chunks])
    records = build_records(chunks, vectors)
    index = get_index()
    for i in range(0, len(records), batch_size):
        index.upsert(vectors=records[i : i + batch_size])
    return len(records)

def search(query: str, top_k: int = 5, company: str | None = None, year: int | None = None):
    query_vector = embed_texts([query])[0]
    flt = {}
    if company:
        flt["company"] = company
    if year:
        flt["year"] = year
    result = get_index().query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        filter=flt or None,
    )
    return result.matches