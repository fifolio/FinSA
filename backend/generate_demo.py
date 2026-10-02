from app.vector_store import search
from app.generation import generate_answer

question = "What are the main risks from supply chain disruption?"
matches = search(question, top_k=3)
chunks = [m.metadata for m in matches]

answer = generate_answer(question, chunks)
print(answer)