NLP‑Based Smart Text API

A fast and simple NLP service built with FastAPI and Hugging Face Transformers. Analyze sentiment, summarize text, and search through documents using semantic similarity.

Getting Started

 Local Setup

1. Install dependencies:

pip install -r requirements.txt
python -m spacy download en_core_web_sm


2. Run the server:

python app.py


Open http://localhost:8000/docs to test the API.

 Docker

Build and run in Docker:

docker build -t text-intelligence-api .
docker run -p 8000:8000 text-intelligence-api


Note: The first run downloads models (~1.5GB) and may take a few minutes.

 API Endpoints

   Analyze Text
Extract sentiment and keywords from text.

curl -X POST http://localhost:8000/analyze \
  -H 'Content-Type: application/json' \
  -d '{"text": "I love working with AI! It makes everything efficient."}'


   Summarize
Generate a concise summary of longer text.

curl -X POST http://localhost:8000/summarize \
  -H 'Content-Type: application/json' \
  -d '{"text": "FastAPI is a modern web framework for building APIs with Python 3.8+..."}'


    Semantic Search
Find similar documents in the vector database.

curl -X POST http://localhost:8000/semantic-search \
  -H 'Content-Type: application/json' \
  -d '{"query": "coding tools", "top_k": 2}'


   Vector Database
Add text: `POST /add-text`  
Delete text: `DELETE /delete-text/{text}`

   Testing
pytest test_app.py

   CI/CD
GitHub Actions runs linting, tests, and Docker builds on every push.

   What's Inside

- Sentiment Analysis: distilbert-base-uncased
- Text Summarization: distilbart-cnn-12-6
- Embeddings: all-MiniLM-L6-v2 (384 dimensions)
- Vector Search: FAISS

Notes

- First run downloads models (~1.5GB)
- In-memory database resets on restart
- Requires: app.py, requirements.txt, Dockerfile
