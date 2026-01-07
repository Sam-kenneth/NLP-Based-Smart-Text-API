import numpy as np
import spacy
import faiss
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from fastapi.responses import HTMLResponse

app = FastAPI(
    title="AI-Powered Text Intelligence API",
    description="NLP service for sentiment, keywords, summarization, and semantic search.",
    version="Latest"
)


try:
    sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    summarizer_pipe = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    nlp = spacy.load("en_core_web_sm")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')

    text_database: List[str] = [
        "FastAPI is a modern, fast web framework for building APIs with Python.",
        "Machine learning models can be deployed efficiently using Docker containers.",
        "Natural Language Processing allows computers to understand human text."
    ]

    dimension = 384
    faiss_index = faiss.IndexFlatL2(dimension)
    initial_embeddings = embedder.encode(text_database).astype('float32')
    faiss_index.add(initial_embeddings)

except Exception as e:
    raise RuntimeError(f"Error loading models: {e}")


class TextRequest(BaseModel):
    text: str = Field(..., min_length=5, example="I love working with AI!")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, example="How to build APIs?")
    top_k: int = Field(default=2, ge=1, le=5)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Text Intelligence API</title>
            <style>
                body { font-family: Arial, sans-serif; display: flex; justify-content: center; 
                       align-items: center; height: 100vh; margin: 0; background-color: #f4f7f6; }
                .container { text-align: center; padding: 50px; background: white; 
                            border-radius: 10px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); }
                h1 { color: #2c3e50; }
                .button { background-color: #007bff; color: white; padding: 15px 25px; 
                          text-decoration: none; border-radius: 5px; font-weight: bold; 
                          display: inline-block; margin-top: 20px; transition: 0.3s; }
                .button:hover { background-color: #0056b3; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>NLP‑Based Smart Text API</h1>
                <p>Hey assessor, please click the button to view my assessment.-Thanks,S.Sam kenneth</p>
                <a href="/docs" class="button">Explore API Swagger UI</a>
            </div>
        </body>
    </html>
    """


@app.post("/analyze")
async def analyze_text(request: TextRequest):
    try:
        sentiment_result = sentiment_pipe(request.text)[0]
        doc = nlp(request.text)
        
        keywords = [chunk.text for chunk in doc.noun_chunks]
        keywords += [token.text for token in doc if token.pos_ == "ADJ"]
        unique_keywords = list(dict.fromkeys(keywords))[:5]

        return {
            "sentiment": sentiment_result['label'].lower(),
            "confidence": round(sentiment_result['score'], 4),
            "keywords": unique_keywords
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")


@app.post("/summarize")
async def summarize_text(request: TextRequest):
    if len(request.text) < 50:
        raise HTTPException(status_code=400, detail="Text too short to summarize.")
    try:
        summary = summarizer_pipe(
            request.text,
            max_length=min(len(request.text.split()) // 2, 150),
            min_length=25,
            do_sample=False
        )
        return {"summary": summary[0]['summary_text']}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")


@app.post("/semantic-search")
async def semantic_search(request: SearchRequest):
    try:
        query_vector = embedder.encode([request.query]).astype('float32')
        distances, indices = faiss_index.search(query_vector, request.top_k)
        results = [text_database[i] for i in indices[0] if i != -1]
        return {"query": request.query, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@app.post("/add-text")
async def add_text(request: TextRequest):
    try:
        vector = embedder.encode([request.text]).astype('float32')
        faiss_index.add(vector)
        text_database.append(request.text)
        return {"message": "Text added successfully", "total_texts": len(text_database)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add text: {e}")


@app.delete("/delete-text/{text}")
async def delete_text(text: str):
    try:
        if text not in text_database:
            raise HTTPException(status_code=404, detail="Text not found in database.")
        idx = text_database.index(text)
        text_database.pop(idx)
        faiss_index.reset()
        embeddings = embedder.encode(text_database).astype('float32')
        faiss_index.add(embeddings)
        return {"message": "Text deleted and index rebuilt", "total_texts": len(text_database)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete text: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
