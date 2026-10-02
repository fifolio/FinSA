from unittest.mock import patch
from app.generation import generate_answer

def test_generate_answer_builds_tagged_context_and_extracts_answer():
    chunks = [
        {"company": "Apple", "year": 2024, "text": "Supply chain risk is elevated."},
        {"company": "Apple", "year": 2024, "text": "China demand softened."},
    ]

    fake_llm = type(
        "FakeLLM",
        (),
        {"invoke": lambda self, prompt: prompt + "Answer: Supply chain risk is the main concern."} 
    )()

    with patch("app.generation.get_llm", return_value=fake_llm):
        answer = generate_answer("What are the risks?", chunks)

    assert answer == "Supply chain risk is the main concern."

def test_generate_answer_includes_company_and_year_tags_in_prompt():
    chunks = [{"company": "Apple", "year": 2024, "text": "Revenue declined."}]
    captured = {}

    def fake_invoke(self, prompt):
        captured["prompt"] = prompt
        return prompt + "Answer: ok"

    fake_llm = type("FakeLLM", (), {"invoke": fake_invoke})()

    with patch("app.generation.get_llm", return_value=fake_llm):
        generate_answer("question", chunks)

    assert "Apple" in captured["prompt"]
    assert "2024" in captured["prompt"]