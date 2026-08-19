# PrepPilot AI

**AI-powered Resume & Job Description Analyzer using RAG**

PrepPilot AI analyzes a candidate's resume against a job description using semantic retrieval and an LLM-based analysis pipeline.

It extracts the resume, breaks it into chunks, generates embeddings, stores them in FAISS, retrieves the most relevant resume sections using the job description, and passes the retrieved context to Gemini to generate a structured recruiter-style report.

##  Live Demo

**Live Application:**  
https://preppilot-ai-9upi.onrender.com/

**Health Check:**  
https://preppilot-ai-9upi.onrender.com/health

**GitHub Repository:**  
https://github.com/Shibbu001/PREPPILOT_AI_V1.0

> The application is deployed as a FastAPI web service on Render.
> The free Render instance may take some time to wake up after inactivity.

---

##  What PrepPilot AI Does

The application allows a user to:

1. Upload a resume PDF.
2. Paste a job description.
3. Extract text from the resume.
4. Split the resume into smaller chunks.
5. Generate local semantic embeddings.
6. Store embeddings in a FAISS vector store.
7. Use the job description as a semantic retrieval query.
8. Retrieve the most relevant resume sections.
9. Pass the retrieved context and job description to Gemini.
10. Generate a structured recruiter-style analysis.

The final report includes:

- Resume match score
- Matching skills
- Missing / weak skills
- Project relevance
- Interview focus questions
- Recommendations

---

#  Architecture

```text
                         USER
                           │
                           ▼
                  ┌─────────────────┐
                  │  HTML/CSS/JS    │
                  │    Frontend     │
                  └────────┬────────┘
                           │
                     POST /analyze
                           │
                           ▼
                  ┌─────────────────┐
                  │     FastAPI     │
                  │    main.py      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │   PDF Parser    │
                  │      pypdf      │
                  └────────┬────────┘
                           │
                      Resume Text
                           │
                           ▼
             ┌──────────────────────────┐
             │ Recursive Character      │
             │ Text Splitter            │
             └────────────┬─────────────┘
                          │
                       Chunks
                          │
                          ▼
             ┌──────────────────────────┐
             │ HuggingFace              │
             │ Sentence Transformers    │
             │ Embeddings               │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │          FAISS           │
             │     Vector Store         │
             └────────────┬─────────────┘
                          │
                  Similarity Search
                  using Job Description
                          │
                          ▼
             ┌──────────────────────────┐
             │ Relevant Resume Context  │
             └────────────┬─────────────┘
                          │
                          ▼
             ┌──────────────────────────┐
             │          Gemini          │
             │       LLM Generation     │
             └────────────┬─────────────┘
                          │
                          ▼
                  Structured JSON
                          │
                          ▼
                    Frontend Report
