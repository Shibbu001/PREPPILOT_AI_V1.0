# PrepPilot AI

A deliberately small, understandable RAG project for resume-vs-job-description analysis.

## What it does

1. User uploads a resume PDF.
2. FastAPI extracts the text.
3. The resume is split into chunks.
4. A local Sentence-Transformers model converts chunks into vectors.
5. FAISS stores the vectors and performs semantic similarity search.
6. The job description is used as the retrieval query.
7. The most relevant resume chunks are passed to Gemini.
8. Gemini produces a recruiter-style analysis.

## Architecture

Resume PDF
    ↓
FastAPI
    ↓
PDF text extraction
    ↓
Chunking
    ↓
Local Embeddings (Sentence Transformers)
    ↓
FAISS Vector Store
    ↓
Similarity Retrieval using Job Description
    ↓
Relevant Resume Context
    ↓
Gemini
    ↓
Match / Skills / Gaps / Interview Focus

## Setup

Use Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your Google API key:

```text
GOOGLE_API_KEY=your_key
```

Google's current Gemini documentation recommends environment variables for API keys.
Do NOT commit `.env` to GitHub.

Run:

```powershell
uvicorn app.main:app --reload
```

Open:

http://127.0.0.1:8000

## Important limitation

This is an MVP, not a production ATS. It intentionally keeps the architecture small so every component can be understood and defended.

The current score field is not calculated by a deterministic formula; Gemini produces the qualitative assessment. A future version could add a deterministic skill-overlap score and structured JSON output.

## Interview explanation

**What:** A resume-vs-JD analyzer.

**How:** PDF → chunks → embeddings → FAISS retrieval → Gemini generation.

**Why RAG:** Instead of sending the entire resume blindly to the LLM, retrieve the most relevant resume sections based on the job description and provide those as context.

**Why FAISS:** It provides local vector similarity search and is simple for an MVP.

**Why LangChain:** It provides reusable abstractions for embeddings, documents, vector stores and LLM integration.

**Why FastAPI:** Lightweight Python API layer for exposing the analysis workflow.

**Why Gemini:** Used as the generation model after retrieval.

## Questions to understand before discussing this project

- What is an embedding?
- What is semantic similarity?
- What is chunking?
- What does FAISS store?
- What happens during similarity_search()?
- What is RAG?
- Why not just send the whole resume to Gemini?
- What is hallucination?
- What is the difference between retrieval and generation?
- Why FastAPI?
- Why use LangChain?
- What would you improve in version 2?
