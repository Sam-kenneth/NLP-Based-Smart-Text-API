from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "Explore API Swagger UI" in response.text

def test_analyze_sentiment():
    response = client.post("/analyze", json={"text": "I love working with AI!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert data["sentiment"] == "positive"