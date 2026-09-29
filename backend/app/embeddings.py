import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", token=hf_token)
    return _model   

def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()