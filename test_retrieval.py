from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

PDF_PATH = "Shivendra Pratap Singh.pdf"

JOB_DESCRIPTION = """
We are looking for highly driven AI/ML Interns who want hands-on exposure
to building agent-based AI systems, RAG pipelines, and production-grade
GenAI applications.

Strong foundation in Python and ML fundamentals.
Built at least one working RAG or agent-based system independently.
Familiar with LangChain, LangGraph or similar frameworks.
Deployed at least one project with GitHub and preferably a live demo.
Able to explain design decisions confidently.
Strong CS fundamentals including DSA and problem solving.
"""

# 1. Extract text from PDF
reader = PdfReader(PDF_PATH)

resume_text = ""

for page in reader.pages:
    resume_text += page.extract_text() + "\n"

print("\n========== EXTRACTED RESUME ==========\n")
print(resume_text)

# 2. Split resume into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

chunks = splitter.split_text(resume_text)

print("\n========== CHUNK COUNT ==========")
print(len(chunks))

# 3. Convert chunks into Documents
documents = [
    Document(page_content=chunk)
    for chunk in chunks
]

# 4. Create local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Store vectors in FAISS
vector_store = FAISS.from_documents(
    documents,
    embeddings
)

# 6. Retrieve relevant resume chunks
results = vector_store.similarity_search(
    JOB_DESCRIPTION,
    k=min(4, len(documents))
)

print("\n========== RETRIEVED CHUNKS ==========\n")

for i, doc in enumerate(results, start=1):
    print(f"\n--- RESULT {i} ---\n")
    print(doc.page_content)

print("\n======================================")
print("RAG retrieval test completed.")