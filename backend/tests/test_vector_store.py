from langchain_core.documents import Document
from app.vector_store import build_records

def test_build_records_shape_and_ids():
    chunks = [
        Document(
            page_content="Risk text",
            metadata={"company": "XYZ", "year": 2025, "source": "t.pdf", "chunk_id": 3},
        )
    ]

    records = build_records(chunks, [[0.1, 0.2]])

    assert records[0]["id"] == "XYZ-2025-3"
    assert records[0]["values"] == [0.1, 0.2]
    assert records[0]["metadata"]["text"] == "Risk text"
    assert records[0]["metadata"]["page"] == -1
