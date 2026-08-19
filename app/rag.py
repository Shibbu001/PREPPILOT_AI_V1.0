
import json
import os
import re

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


# ============================================================
# HELPERS
# ============================================================

def _extract_response_text(response) -> str:
    """
    Gemini/LangChain can return response.content as either
    a string or a list of content blocks.

    Normalize everything into a plain string.
    """

    content = response.content

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):
                # Common LangChain content-block format
                if "text" in item:
                    parts.append(str(item["text"]))

        return "\n".join(parts).strip()

    return str(content).strip()


def _parse_json_response(response_text: str) -> dict:
    """
    Safely parse Gemini's JSON response.

    Handles:
    - normal JSON
    - ```json ... ``` responses
    - accidental surrounding text
    """

    cleaned = response_text.strip()

    # Remove Markdown code fences if Gemini adds them.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    cleaned = cleaned.strip()

    # First attempt: direct JSON parsing.
    try:
        result = json.loads(cleaned)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Second attempt: find the outermost JSON object.
    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start != -1 and end != -1 and end > start:

        json_candidate = cleaned[start:end + 1]

        try:
            result = json.loads(json_candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    raise RuntimeError(
        "Gemini returned an invalid JSON response.\n\n"
        f"Raw response:\n{response_text}"
    )


# ============================================================
# MAIN ANALYZER
# ============================================================

def analyze_resume(
    resume_text: str,
    job_description: str
):

    # ========================================================
    # 1. API KEY CHECK
    # ========================================================

    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GOOGLE_API_KEY is not set."
        )


    # ========================================================
    # 2. VALIDATE INPUT
    # ========================================================

    if not resume_text or not resume_text.strip():
        raise ValueError(
            "Resume text is empty."
        )

    if not job_description or not job_description.strip():
        raise ValueError(
            "Job description is empty."
        )


    # ========================================================
    # 3. SPLIT RESUME INTO CHUNKS
    # ========================================================

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(
        resume_text
    )

    if not chunks:
        raise ValueError(
            "Resume text could not be split into chunks."
        )

    documents = [
        Document(page_content=chunk)
        for chunk in chunks
    ]


    # ========================================================
    # 4. CREATE LOCAL EMBEDDINGS
    # ========================================================

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


    # ========================================================
    # 5. CREATE FAISS VECTOR STORE
    # ========================================================

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )


    # ========================================================
    # 6. RETRIEVE RELEVANT RESUME SECTIONS
    # ========================================================

    relevant_docs = vector_store.similarity_search(
        job_description,
        k=min(6, len(documents))
    )

    if not relevant_docs:
        raise ValueError(
            "Could not retrieve relevant resume sections."
        )

    context = "\n\n---\n\n".join(
        doc.page_content
        for doc in relevant_docs
    )


    # ========================================================
    # 7. INITIALIZE GEMINI
    # ========================================================

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite"
    )


    # ========================================================
    # 8. RECRUITER ANALYSIS PROMPT
    # ========================================================

    prompt = f"""
You are PrepPilot AI, an intelligent technical recruitment
assistant.

Your task is to evaluate a candidate's resume against a
specific job description.

IMPORTANT:

Use ONLY information contained in the retrieved resume
context.

Do NOT invent:
- skills
- projects
- experience
- deployments
- certifications
- technologies
- achievements
- work history

If something is merely listed as a skill but there is no
project or evidence supporting it, treat it as weaker evidence.

If a requirement is not demonstrated in the resume context,
explicitly mark it as missing or weak.

Do not judge the candidate based on assumptions.

============================================================
JOB DESCRIPTION
============================================================

{job_description}


============================================================
RETRIEVED RESUME CONTEXT
============================================================

{context}


============================================================
ANALYSIS REQUIREMENTS
============================================================

Evaluate the candidate across:

1. Overall match
2. Matching technical skills
3. Missing or weak requirements
4. Project relevance
5. Interview risks / focus areas
6. Concrete recommendations


============================================================
OUTPUT FORMAT
============================================================

Return ONLY valid JSON.

Use EXACTLY this structure:

{{
    "match_score": 0,

    "summary": "",

    "matching_skills": [
        {{
            "skill": "",
            "evidence": "",
            "strength": "strong"
        }}
    ],

    "missing_skills": [
        {{
            "skill": "",
            "reason": "",
            "priority": "high"
        }}
    ],

    "project_relevance": [
        {{
            "project": "",
            "relevance": "",
            "score": 0
        }}
    ],

    "interview_focus": [
        "",
        "",
        ""
    ],

    "recommendations": [
        "",
        ""
    ]
}}


============================================================
STRICT RULES
============================================================

match_score:
- Integer from 0 to 100.
- Base the score on actual evidence.
- Do not give a high score merely because keywords appear.

summary:
- 1-3 concise sentences.
- Explain the overall candidate-job fit.

matching_skills:
- Include 3-6 skills.
- Only include skills actually supported by the resume.
- "evidence" must explain where/how they are demonstrated.
- strength must be one of:
  "strong", "moderate", "weak"

missing_skills:
- Include 3-6 important missing or weak requirements.
- Prioritize requirements explicitly mentioned in the JD.
- priority must be one of:
  "high", "medium", "low"

project_relevance:
- Include 2-3 relevant projects from the resume.
- score must be an integer from 0 to 10.
- Explain why each project is or is not relevant.

interview_focus:
- EXACTLY 3 questions.
- Questions should target genuine gaps, important projects,
  or important requirements from the JD.

recommendations:
- Include 2-4 actionable recommendations.
- Recommendations should tell the candidate what they should
  improve, build, learn, demonstrate, or add to the resume.

IMPORTANT:
- Do not use Markdown.
- Do not add text before the JSON.
- Do not add text after the JSON.
- Return valid JSON only.
"""


    # ========================================================
    # 9. CALL GEMINI
    # ========================================================

    response = llm.invoke(
        prompt
    )


    # ========================================================
    # 10. NORMALIZE GEMINI RESPONSE
    # ========================================================

    response_text = _extract_response_text(
        response
    )

    if not response_text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )


    # ========================================================
    # 11. PARSE JSON
    # ========================================================

    report = _parse_json_response(
        response_text
    )


    # ========================================================
    # 12. BASIC OUTPUT VALIDATION
    # ========================================================

    required_keys = [
        "match_score",
        "summary",
        "matching_skills",
        "missing_skills",
        "project_relevance",
        "interview_focus",
        "recommendations"
    ]

    for key in required_keys:

        if key not in report:
            raise RuntimeError(
                f"Gemini response is missing required field: {key}"
            )


    # Keep score within the expected range.
    try:
        report["match_score"] = int(
            report["match_score"]
        )
    except (TypeError, ValueError):
        report["match_score"] = 0

    report["match_score"] = max(
        0,
        min(100, report["match_score"])
    )


    # ========================================================
    # 13. RETURN RESULT
    # ========================================================

    return {
        "report": report,
        "retrieved_chunks": relevant_docs,
        "chunk_count": len(chunks)
    }