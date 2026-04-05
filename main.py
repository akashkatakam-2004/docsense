from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid, os

from rag import process_pdf, ask_question, stream_answer, clear_session, is_model_ready, load_saved_indexes

load_dotenv()

def validate_env():
    key = os.getenv("GROQ_API_KEY")
    if not key or key.strip() == "":
        print("ERROR: GROQ_API_KEY is missing!")
        raise SystemExit(1)

validate_env()

app = FastAPI(title="DocSense RAG")
app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = "uploads"
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
os.makedirs(UPLOAD_DIR, exist_ok=True)

class QuestionRequest(BaseModel):
    session_id: str
    question: str
    model: str = "llama-3.3-70b-versatile"

@app.on_event("startup")
async def startup_event():
    load_saved_indexes()

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/api/upload")
async def upload_pdf(request: Request, file: UploadFile = File(...), model: str = "llama-3.3-70b-versatile"):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB.")
    if len(contents) < 100:
        raise HTTPException(status_code=400, detail="File is too small or empty.")
    session_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")
    with open(file_path, "wb") as f:
        f.write(contents)
    result = process_pdf(file_path, session_id, model)
    if not result["success"]:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=result["error"])
    return {"session_id": session_id, "filename": file.filename, "pages": result["pages"], "chunks": result["chunks"], "message": f"Ready! {result['pages']} pages indexed."}

@app.post("/api/ask")
async def ask(req: QuestionRequest, request: Request):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    result = ask_question(req.session_id, req.question)
    if not result["success"]:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"answer": result["answer"], "sources": result["sources"]}

@app.get("/api/stream")
async def stream(request: Request, session_id: str, question: str):
    return StreamingResponse(stream_answer(session_id, question), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.get("/api/ready")
async def ready():
    return {"ready": is_model_ready()}

@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    clear_session(session_id)
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}.pdf")
    if os.path.exists(file_path):
        os.remove(file_path)
    return {"message": "Session cleared."}

@app.get("/api/models")
async def get_models():
    return {"models": ["llama-3.3-70b-versatile","llama-3.1-8b-instant","mixtral-8x7b-32768","gemma2-9b-it"]}

@app.get("/api/health")
async def health():
    return {"status": "ok", "groq_key_set": bool(os.getenv("GROQ_API_KEY")), "model_ready": is_model_ready(), "upload_dir": os.path.exists(UPLOAD_DIR)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
