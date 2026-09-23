# main.py
# FastAPI wrapper — turns the RAG system into an interactive API.
# Run with:  uvicorn main:app --reload
# Then open: http://127.0.0.1:8000/docs

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_system import SimpleRAG

app = FastAPI(title="RAG Q&A API")
rag = SimpleRAG()


# --- Request/Response schemas ---

class AddDocsRequest(BaseModel):
    documents: list[str]

class QuestionRequest(BaseModel):
    question: str
    top_k: int = 3

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]


# --- Endpoints ---

@app.post("/index")
def index_documents(request: AddDocsRequest):
    """Add documents to the knowledge base."""
    rag.add_documents(request.documents)
    return {"status": "indexed", "count": len(request.documents)}

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """Ask a question — retrieves relevant context and generates an answer."""
    try:
        result = rag.generate_answer(request.question, top_k=request.top_k)
        return AnswerResponse(answer=result["answer"], sources=result["sources"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"status": "RAG API is running", "docs": "/docs"}
