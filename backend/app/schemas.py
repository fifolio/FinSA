from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None

class AskResponse(BaseModel):
    answer: str
    sources: list[dict]

class UploadResponse(BaseModel):
    chunks_upserted: int
    company: str
    year: int

