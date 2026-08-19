from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader
from app.rag import analyze_resume


PDF_PATH = "Shivendra Pratap Singh.pdf"

JOB_DESCRIPTION = """
CAW Studios / KnackLabs is hiring for an Associate Software Engineer Intern.

The company is looking for students with:
- Strong CS fundamentals and problem-solving skills
- DSA knowledge
- OOP and SOLID principles
- AI/ML fundamentals
- Understanding of agentic systems
- RAG pipelines
- LLM fine-tuning
- Real-time project experience
- Ability to explain technical projects and design decisions

The role involves building product engineering solutions and working with
AI/ML and modern software engineering technologies.
"""


# Extract resume text
reader = PdfReader(PDF_PATH)

resume_text = ""

for page in reader.pages:
    text = page.extract_text()

    if text:
        resume_text += text + "\n"


print("Resume extracted.")
print("Characters:", len(resume_text))

print("\nRunning PrepPilot AI...")
print("This may take a little while.\n")


result = analyze_resume(
    resume_text=resume_text,
    job_description=JOB_DESCRIPTION
)


print("\n==============================")
print("       PREPPILOT AI REPORT")
print("==============================\n")

print(result["report"])

print("\n==============================")
print("Retrieved chunks:", len(result["retrieved_chunks"]))
print("Total chunks:", result["chunk_count"])
print("==============================")