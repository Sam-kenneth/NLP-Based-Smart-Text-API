from setuptools import setup, find_packages

setup(
    name="NLP‑Based Smart Text API",
    version="Latest",
    description="An AI-Powered Text Intelligence API with FastAPI and Hugging Face",
    author="S.Sam kenneth",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "fastapi",
        "uvicorn[standard]",
        "pydantic",
        "transformers",
        "torch",
        "spacy",
        "sentence-transformers",
        "faiss-cpu",
        "numpy",
        "httpx",  
        "pytest"
    ],
    extras_require={
        "dev": ["flake8", "pytest-asyncio"],
    },
)