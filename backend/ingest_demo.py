from app.ingestion import ingest_pdf

chunks = ingest_pdf("data/apple_2025.pdf", company="Apple", year=2025)

print(f"Total chunks: {len(chunks)}")

for chunk in chunks[:3]:
    print("-" * 60)
    print(chunk.metadata)
    print(chunk.page_content[:300])