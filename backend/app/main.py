import shutil
import tempfile
from pathlib import Path


from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.ingestion import ingest_pdf
from app.vector_store import upsert_chunks, search
from app.generation import generate_answer
from app.schemas import AskRequest, AskResponse, UploadResponse


app = FastAPI(title="FinSA - Financial RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile, company: str, year: int):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

        try:
            chunks = ingest_pdf(tmp_path, company=company, year=year)
            count = upsert_chunks(chunks)
        finally:
            Path(tmp_path).unlink()

        return UploadResponse(chunks_upserted=count, company=company, year=year)

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    matches = search(request.question, top_k=5, company=request.company, year=request.year)
    if not matches:
        raise HTTPException(status_code=404, detail="No relevant documents found.")

    chunks = [m.metadata for m in matches]
    answer = generate_answer(request.question, chunks)
    return AskResponse(answer=answer, sources=chunks)