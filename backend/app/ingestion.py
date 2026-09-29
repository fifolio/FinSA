from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_pdf(path: str, company: str, year: int) -> list[Document]:
    """Load a PDF as one Document per page, tagged with metadata."""
    pages = PyPDFLoader(str(path)).load()
    for page in pages:
        page.metadata.update(
            {"company": company, "year": year, "source": Path(path).name}
        )
    return pages

def chunk_documents(docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    """Split documents into overlapping chunks and number them."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size , chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks

def ingest_pdf(path: str, company: str, year: int) -> list[Document]:
    """Load a PDF, split it into chunks, and return the chunks."""
    return chunk_documents(load_pdf(path, company, year))