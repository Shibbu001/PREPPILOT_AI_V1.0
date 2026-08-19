from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from pypdf import PdfReader
from io import BytesIO
import os

from app.rag import analyze_resume


app = FastAPI(
    title="PrepPilot AI",
    description="AI-powered resume and job description analyzer",
    version="1.0.0"
)


# ==========================================
# HOME
# ==========================================

@app.get("/")
def root():
    return FileResponse("static/index.html")


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "PrepPilot AI"
    }


# ==========================================
# RESUME ANALYSIS
# ==========================================

@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):

    # --------------------------------------
    # 1. Validate PDF
    # --------------------------------------

    if not resume.filename:
        raise HTTPException(
            status_code=400,
            detail="No resume file provided."
        )

    if not resume.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF resumes are supported."
        )


    # --------------------------------------
    # 2. Read PDF
    # --------------------------------------

    file_bytes = await resume.read()

    try:
        reader = PdfReader(BytesIO(file_bytes))

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Could not read the PDF file."
        )


    # --------------------------------------
    # 3. Extract resume text
    # --------------------------------------

    resume_text = ""

    for page in reader.pages:

        text = page.extract_text()

        if text:
            resume_text += text + "\n"


    if not resume_text.strip():
        raise HTTPException(
            status_code=400,
            detail="Could not extract text from the resume."
        )


    # --------------------------------------
    # 4. Validate JD
    # --------------------------------------

    if not job_description.strip():
        raise HTTPException(
            status_code=400,
            detail="Job description cannot be empty."
        )


    # --------------------------------------
    # 5. Run RAG + Gemini
    # --------------------------------------

    try:

        result = analyze_resume(
            resume_text=resume_text,
            job_description=job_description
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Resume analysis failed: {str(e)}"
        )


    # --------------------------------------
    # 6. Return structured response
    # --------------------------------------

    return {
        "filename": resume.filename,
        "report": result["report"],
        "retrieved_chunks": [
            doc.page_content
            for doc in result["retrieved_chunks"]
        ],
        "chunk_count": result["chunk_count"]
    }