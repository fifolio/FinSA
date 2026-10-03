from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

_llm = None

PROMPT_TEMPLATE = """
You are a financial analyst assistant. Answer the question using only the context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:
"""

def get_llm() -> HuggingFacePipeline:
    global _llm
    if _llm is None:
        hf_pipeline = pipeline(
            "text-generation",
            model="HuggingFaceTB/SmolLM2-135M-Instruct",
            max_new_tokens=300
        )
        _llm = HuggingFacePipeline(pipeline=hf_pipeline)
    return _llm

def generate_answer(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{c['company']} {c['year']}] {c['text']}" for c in chunks
    )
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.split("Answer:")[-1].strip()