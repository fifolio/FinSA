from langchain_core.documents import Document

from app.ingestion import chunk_documents

def test_chunking_splits_and_keeps_metadata(): 
    doc = Document(
        page_content = "Risk factor. " * 500,
        metadata = {"company": "XYZ", "year": 2025, "source": "test.pdf"} 
    )

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.metadata["company"] == "XYZ" for c in chunks)
    assert [c.metadata["chunk_id"] for c in chunks] == list(range(len(chunks)))
    assert all(len(c.page_content) <= 200 for c in chunks)