from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os, time, threading

load_dotenv()

vector_stores = {}
chains        = {}
last_access   = {}
model_ready   = False   # FIX 2: first-time loading flag

FAISS_DIR = "faiss_indexes"  # FIX 3: disk persistence folder
os.makedirs(FAISS_DIR, exist_ok=True)

# ── FIX 3: load existing FAISS indexes from disk on startup ──
def load_saved_indexes():
    loaded = 0
    if not os.path.exists(FAISS_DIR):
        return
    embeddings = get_embeddings()
    for name in os.listdir(FAISS_DIR):
        sid = name  # folder name = session_id
        idx_path = os.path.join(FAISS_DIR, sid)
        if os.path.isdir(idx_path):
            try:
                vs = FAISS.load_local(idx_path, embeddings, allow_dangerous_deserialization=True)
                vector_stores[sid] = vs
                last_access[sid] = time.time()
                loaded += 1
            except Exception as e:
                print(f"[startup] Could not load index {sid}: {e}")
    if loaded:
        print(f"[startup] Restored {loaded} FAISS index(es) from disk")

# ── AUTO CLEANUP every 5 min ──
def cleanup_old_sessions():
    while True:
        time.sleep(300)
        now = time.time()
        expired = [sid for sid, t in list(last_access.items()) if now - t > 1800]
        for sid in expired:
            vector_stores.pop(sid, None)
            chains.pop(sid, None)
            last_access.pop(sid, None)
            # Remove FAISS from disk too
            idx_path = os.path.join(FAISS_DIR, sid)
            if os.path.exists(idx_path):
                import shutil
                shutil.rmtree(idx_path)
            fpath = os.path.join(os.getenv("UPLOAD_DIR", "uploads"), f"{sid}.pdf")
            if os.path.exists(fpath):
                os.remove(fpath)
            print(f"[cleanup] Removed expired session: {sid}")

cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
cleanup_thread.start()


def get_embeddings():
    global model_ready
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        model_ready = True
        return emb
    except Exception:
        pass
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        model_ready = True
        return emb
    except Exception:
        pass
    import hashlib, struct
    from langchain.embeddings.base import Embeddings
    from typing import List
    class FallbackEmbeddings(Embeddings):
        def embed_documents(self, texts: List[str]) -> List[List[float]]:
            return [self._embed(t) for t in texts]
        def embed_query(self, text: str) -> List[float]:
            return self._embed(text)
        def _embed(self, text: str) -> List[float]:
            result = []
            for i in range(384):
                h = hashlib.md5(f"{text}{i}".encode()).digest()
                result.append(float(struct.unpack('f', h[:4])[0]))
            return result
    print("[warning] Using fallback embeddings")
    model_ready = True
    return FallbackEmbeddings()


def get_llm(model="llama-3.3-70b-versatile"):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing. Please check your .env file.")
    return ChatGroq(model=model, api_key=api_key, temperature=0.3, max_retries=3)


def is_model_ready():
    return model_ready


def process_pdf(pdf_path: str, session_id: str, model: str = "llama-3.3-70b-versatile") -> dict:
    try:
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        if not documents:
            return {"success": False, "error": "Could not read PDF. File may be empty or corrupted."}

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        if len(chunks) == 0:
            return {"success": False, "error": "PDF has no readable text content."}

        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # FIX 3: save FAISS index to disk
        idx_path = os.path.join(FAISS_DIR, session_id)
        vectorstore.save_local(idx_path)

        vector_stores[session_id] = vectorstore

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=get_llm(model),
            retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
            memory=memory,
            return_source_documents=True,
            verbose=False,
        )
        chains[session_id] = chain
        last_access[session_id] = time.time()
        return {"success": True, "pages": len(documents), "chunks": len(chunks)}

    except ValueError as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": f"Processing failed: {str(e)}"}


def ask_question(session_id: str, question: str) -> dict:
    if session_id not in chains:
        # Try to rebuild chain from saved FAISS index
        if session_id in vector_stores:
            try:
                memory = ConversationBufferMemory(
                    memory_key="chat_history", return_messages=True, output_key="answer"
                )
                chain = ConversationalRetrievalChain.from_llm(
                    llm=get_llm(),
                    retriever=vector_stores[session_id].as_retriever(search_kwargs={"k": 4}),
                    memory=memory,
                    return_source_documents=True,
                    verbose=False,
                )
                chains[session_id] = chain
            except Exception as e:
                return {"success": False, "error": f"Could not restore session: {str(e)}"}
        else:
            return {"success": False, "error": "Session expired or PDF not uploaded. Please upload your PDF again."}
    try:
        last_access[session_id] = time.time()
        result = chains[session_id].invoke({"question": question})
        answer = result.get("answer", "")
        sources = []
        for doc in result.get("source_documents", []):
            page = doc.metadata.get("page", 0) + 1
            if page not in sources:
                sources.append(page)
        return {"success": True, "answer": answer, "sources": sorted(sources)}
    except Exception as e:
        return {"success": False, "error": f"Error generating answer: {str(e)}"}


def stream_answer(session_id: str, question: str):
    """FIX 1: Generator that yields answer tokens for SSE streaming"""
    if session_id not in vector_stores:
        yield f"data: ERROR: Session expired. Please upload your PDF again.\n\n"
        return
    try:
        last_access[session_id] = time.time()
        # Get relevant chunks
        vs = vector_stores[session_id]
        docs = vs.similarity_search(question, k=4)
        context = "\n\n".join([d.page_content for d in docs])
        sources = sorted(list(set([d.metadata.get("page", 0) + 1 for d in docs])))

        # Build prompt
        prompt = f"""You are a helpful assistant answering questions about a PDF document.
Use ONLY the following context from the PDF to answer the question.
If the answer is not in the context, say "I could not find this in the document."

Context from PDF:
{context}

Question: {question}

Answer:"""

        # Stream from Groq
        llm = get_llm()
        for chunk in llm.stream(prompt):
            token = chunk.content
            if token:
                # Escape newlines for SSE
                token_escaped = token.replace("\n", "\\n")
                yield f"data: TOKEN:{token_escaped}\n\n"

        # Send sources at end
        yield f"data: SOURCES:{','.join(map(str, sources))}\n\n"
        yield f"data: DONE\n\n"

    except Exception as e:
        yield f"data: ERROR:{str(e)}\n\n"


def clear_session(session_id: str):
    vector_stores.pop(session_id, None)
    chains.pop(session_id, None)
    last_access.pop(session_id, None)
    # Remove from disk
    idx_path = os.path.join(FAISS_DIR, session_id)
    if os.path.exists(idx_path):
        import shutil
        shutil.rmtree(idx_path)
