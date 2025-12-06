That's enough theory, let's build now!

## GitHub Template Repository

[https://github.com/yashas-hm/ask-yashas-llm](https://github.com/yashas-hm/ask-yashas-llm)

---

## Tech Stack

| Component        | Technology                |
|------------------|---------------------------|
| **Backend**      | FastAPI                   |
| **LLM**          | Google Gemini 2.0 Flash   |
| **Embeddings**   | Google text-embedding-004 |
| **Vector Store** | Upstash Vector            |
| **Deployment**   | Vercel                    |
| **CI/CD**        | GitHub Actions            |

---

## Project Structure

```
ask-yashas-llm/
├── api/
│   ├── endpoints/
│   │   ├── answer.py         # POST /api/prompt
│   │   ├── default.py        # GET / (redirect)
│   │   └── health_check.py   # GET /api/healthCheck
│   ├── model/
│   │   └── query_model.py    # Request/response models
│   └── utils/
│       ├── llm_pipeline.py   # RAG pipeline
│       ├── middleware.py     # Security middleware
│       ├── bypass_key_gen.py # Generate bypass keys
│       └── upload_vectorstore_data.py  # Upload to Upstash
├── app.py                    # FastAPI entry point
├── rag_data.json             # Structured data for RAG
├── requirements.txt
├── vercel.json               # Vercel config
└── .github/workflows/
    └── data_change_action.yml
```

---

## Step 1: Set Up Upstash Vector

1. Create a free account at [console.upstash.com](https://console.upstash.com)
2. Create a Vector index with:
    - **Dimensions:** 768 (for text-embedding-004)
    - **Distance Metric:** Cosine
3. Copy the **REST URL** and **Token**

---

## Step 2: Prepare Your Data

Create a structured JSON file (`rag_data.json`) with your information:

```json
{
  "about": {
    "name": "Your Name",
    "title": "Your Title",
    "summary": "Your bio...",
    "highlights": [
      "Highlight 1",
      "Highlight 2"
    ]
  },
  "skills": {
    "domains": [
      "Frontend",
      "Backend"
    ],
    "languages": [
      "Python",
      "JavaScript"
    ],
    "frameworks": [
      "FastAPI",
      "Flutter"
    ]
  },
  "experience": [
    {
      "role": "Software Engineer",
      "company": "Company Name",
      "duration": "Jan 2023 – Present",
      "highlights": [
        "Achievement 1",
        "Achievement 2"
      ]
    }
  ],
  "projects": [
    {
      "name": "Project Name",
      "description": "What it does",
      "skills": [
        "Python",
        "FastAPI"
      ],
      "link": "https://github.com/..."
    }
  ]
}
```

---

## Step 3: Upload Data to Upstash Vector

The upload script chunks your JSON data semantically and uploads embeddings:

```python
"""
Upload embeddings to Upstash Vector from structured JSON data.
"""

import hashlib
import json
import os
import google.generativeai as genai
from upstash_vector import Index

EMBEDDING_MODEL = "models/text-embedding-004"
RAG_DATA_FILE = "rag_data.json"


def load_and_chunk_json(file_path: str) -> list[dict]:
    """Load JSON and create semantic chunks - each section/item becomes a chunk."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    chunks = []

    # About section - single chunk
    if 'about' in data:
        about = data['about']
        text = f"""About {about['name']}
{about['title']}

{about['summary']}

Key highlights:
{chr(10).join('- ' + h for h in about.get('highlights', []))}"""
        chunks.append({"type": "about", "text": text.strip()})

    # Skills - single chunk
    if 'skills' in data:
        skills = data['skills']
        text = f"""Skills

Domains: {', '.join(skills.get('domains', []))}
Languages: {', '.join(skills.get('languages', []))}
Frameworks: {', '.join(skills.get('frameworks', []))}"""
        chunks.append({"type": "skills", "text": text.strip()})

    # Experience - each job is a chunk
    for exp in data.get('experience', []):
        text = f"""Work Experience: {exp['role']} at {exp['company']}
Duration: {exp['duration']}

Achievements:
{chr(10).join('- ' + h for h in exp.get('highlights', []))}"""
        chunks.append({"type": "experience", "text": text.strip()})

    # Projects - each project is a chunk
    for proj in data.get('projects', []):
        text = f"""Project: {proj['name']}

{proj['description']}

Technologies: {', '.join(proj.get('skills', []))}
Link: {proj.get('link', 'N/A')}"""
        chunks.append({"type": "project", "text": text.strip()})

    # Add ID to each chunk
    for chunk in chunks:
        chunk["id"] = hashlib.md5(chunk["text"].encode()).hexdigest()

    return chunks


def get_embeddings(texts: list[str], api_key: str) -> list[list[float]]:
    """Generate embeddings using Google's embedding API."""
    genai.configure(api_key=api_key)

    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type="retrieval_document"
        )
        embeddings.append(result['embedding'])

    return embeddings


def upload_vector_data(chunks: list[dict], embeddings: list[list[float]]):
    """Upload vectors to Upstash."""
    index = Index(
        url=os.environ["UPSTASH_VECTOR_REST_URL"],
        token=os.environ["UPSTASH_VECTOR_REST_TOKEN"]
    )

    # Clear existing data first
    index.reset()

    # Prepare vectors with metadata
    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["id"],
            "vector": embedding,
            "metadata": {"text": chunk["text"], "type": chunk["type"]}
        })

    # Upsert in batches
    index.upsert(vectors=vectors)
    print(f"Uploaded {len(vectors)} vectors to Upstash")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    chunks = load_and_chunk_json(RAG_DATA_FILE)
    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts, os.environ["API_TOKEN"])
    upload_vector_data(chunks, embeddings)
```

Run the upload:

```bash
python api/utils/upload_vectorstore_data.py
```

---

## Step 4: Build the RAG Pipeline

The pipeline embeds queries, searches Upstash, and generates responses with Gemini:

```python
"""
LLM Pipeline for RAG-based question answering.
"""

import os
import google.generativeai as genai
from upstash_vector import Index

LLM_MODEL = 'gemini-2.0-flash'
EMBEDDING_MODEL = "models/text-embedding-004"
BOT = 'AskYashas'
NAME = 'Yashas Majmudar'


def get_prompt(query: str, context: str, history: str) -> str:
    """Generate the prompt for the LLM."""
    return f"""You are {BOT}, a friendly AI assistant on {NAME}'s portfolio website.

Guidelines:
1. Only answer questions about {NAME} or respond to greetings.
2. Be conversational and concise.
3. Base answers strictly on the provided context.
4. Use markdown formatting for readability.

Conversation History:
{history}

Context:
{context}

Question: {query}

Respond in markdown:"""


class LLMPipeline:
    """Serverless RAG pipeline using Upstash Vector + Google Gemini."""

    def __init__(self):
        self._index = None
        self._genai_configured = False

    @property
    def index(self) -> Index:
        """Lazy initialization of Upstash Vector index."""
        if self._index is None:
            self._index = Index(
                url=os.environ["UPSTASH_VECTOR_REST_URL"],
                token=os.environ["UPSTASH_VECTOR_REST_TOKEN"]
            )
        return self._index

    def _ensure_genai(self):
        """Ensure Google GenAI is configured."""
        if not self._genai_configured:
            genai.configure(api_key=os.environ["API_TOKEN"])
            self._genai_configured = True

    def _embed_query(self, query: str) -> list[float]:
        """Embed query using Google's embedding API."""
        self._ensure_genai()
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']

    def _search_similar(self, query_embedding: list[float], top_k: int = 5) -> list[str]:
        """Search Upstash Vector for similar documents."""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [r.metadata["text"] for r in results if r.metadata]

    def _generate_response(self, query: str, context: str, history: str) -> str:
        """Generate response using Gemini."""
        self._ensure_genai()

        model = genai.GenerativeModel(LLM_MODEL)
        response = model.generate_content(
            get_prompt(query, context, history),
            generation_config=genai.GenerationConfig(
                candidate_count=1,
                max_output_tokens=2048,
            )
        )

        if response.candidates and response.candidates[0].content.parts:
            return "".join(part.text for part in response.candidates[0].content.parts)
        return response.text

    def invoke(self, query: str, history: list) -> str:
        """Main entry point: embed query -> search -> generate response."""
        query_embedding = self._embed_query(query)
        similar_docs = self._search_similar(query_embedding)
        context = "\n\n".join(similar_docs)
        history_str = self._format_history(history)
        return self._generate_response(query, context, history_str)

    @staticmethod
    def _format_history(history: list) -> str:
        """Format conversation history."""
        if not history:
            return "No History"
        formatted = []
        for chat in history:
            role = chat.get('role', '').upper()
            message = chat.get('message', '')
            if role == 'AI':
                formatted.append(f'{BOT}: {message}')
            elif role == 'HUMAN':
                formatted.append(f'User: {message}')
        return '\n'.join(formatted) if formatted else "No History"


# Singleton
_pipeline = None


def get_pipeline() -> LLMPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = LLMPipeline()
    return _pipeline
```

---

## Step 5: Serve via FastAPI

```python
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from api.utils.llm_pipeline import get_pipeline


class QueryModel(BaseModel):
    query: str
    history: Optional[list] = []


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.post('/api/prompt')
async def answer_endpoint(query: QueryModel):
    try:
        pipeline = get_pipeline()
        result = pipeline.invoke(query=query.query, history=query.history)
        return JSONResponse(content={"response": result}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/healthCheck')
async def health_check():
    return JSONResponse(content={"status": "healthy"}, status_code=200)
```

---

## Step 6: Deploy on Vercel

### 1. Create `vercel.json`:

```json
{
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ]
}
```

### 2. Deploy:

```bash
npm i -g vercel
vercel
```

### 3. Add environment variables in Vercel dashboard:

- `API_TOKEN`
- `UPSTASH_VECTOR_REST_URL`
- `UPSTASH_VECTOR_REST_TOKEN`

---

## Step 7: Automate with GitHub Actions

Auto-update vectors when `rag_data.json` changes:

```yaml
name: Update Upstash Vector on Data Change

on:
  workflow_dispatch:
  push:
    branches:
      - main
    paths:
      - rag_data.json

jobs:
  update-vectors:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: '3.13'

      - name: Install Dependencies
        run: pip install google-generativeai upstash-vector

      - name: Upload to Upstash Vector
        env:
          API_TOKEN: ${{ secrets.API_TOKEN }}
          UPSTASH_VECTOR_REST_URL: ${{ secrets.UPSTASH_VECTOR_REST_URL }}
          UPSTASH_VECTOR_REST_TOKEN: ${{ secrets.UPSTASH_VECTOR_REST_TOKEN }}
        run: python api/utils/upload_vectorstore_data.py --workflow
```

---

## Performance Benefits

| Metric           | Before                    | After              |
|------------------|---------------------------|--------------------|
| **Cold Start**   | 10-30s                    | ~1s                |
| **Memory**       | 90MB+ (HuggingFace model) | Minimal            |
| **Vector Store** | Local ChromaDB            | Hosted Upstash     |
| **Deployment**   | Railway                   | Vercel (free tier) |

---

## Conclusion

This serverless RAG architecture eliminates the cold start problem by using hosted services (Upstash Vector, Google
APIs) instead of loading ML models locally. You get a fast, scalable, memory-aware chatbot that lives in the cloud and
knows everything about you.
