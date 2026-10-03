from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_ask_returns_answer_and_sources():
    fake_match = MagicMock()
    fake_match.metadata = {"company": "Apple", "year": 2024, "text": "Risk text."}

    with patch("app.main.search", return_value=[fake_match]), \
            patch("app.main.generate_answer", return_value="This is the answer."):
        response = client.post("/ask", json={"question": "What are the risks?"})

        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "This is the answer."
        assert body["sources"][0]["company"] == "Apple"

def test_ask_returns_404_when_no_matches():
    with patch("app.main.search", return_value=[]):
        response = client.post("/ask", json={"question": "Anything?"})

        assert response.status_code == 404

def test_ask_rejects_missing_question():
    response = client.post("/ask", json={})

    assert response.status_code == 422 