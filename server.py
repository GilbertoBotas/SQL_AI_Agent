import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Pydantic serializer warnings")

import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel
from modules.sql_agent import SQLAgent, SQLAgentResponse


# ── One SQLAgent per session ──────────────────────────────────────────────────

sessions: dict[str, SQLAgent] = {}


def get_or_create_session(session_id: str) -> SQLAgent:
    """Returns existing agent for session_id, or creates a new one."""
    if session_id not in sessions:
        agent = SQLAgent()
        agent.initialize()
        sessions[session_id] = agent
    return sessions[session_id]


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("API ready.")
    yield
    sessions.clear()
    print("Sessions cleared. Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NedBank RAG Agent API",
    description="Conversational banking assistant powered by NedBank documents.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class StartSessionResponse(PydanticBaseModel):
    session_id: str


class AskRequest(PydanticBaseModel):
    session_id: str
    question: str

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_id": "abc-123",
                "question": "What is MyUey?"
            }
        }
    }


class AskResponse(PydanticBaseModel):
    session_id: str
    answer: str
    sql: str
    confidence: str
    category: str | None
    assumptions: list[str] | None
    needs_clarification: bool
    clarification_question: str | None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    """Check if the API is running."""
    return {"status": "ok"}


@app.post("/session/start", response_model=StartSessionResponse, tags=["Session"])
def start_session():
    """
    Create a new session and return its session_id.
    Call this once per user before sending any questions.
    """
    session_id = str(uuid.uuid4())
    get_or_create_session(session_id)
    return StartSessionResponse(session_id=session_id)


@app.post("/chat", response_model=AskResponse, tags=["Chat"])
def ask(request: AskRequest) -> AskResponse:
    """
    Ask a question. Pass the session_id returned from /session/start.
    Each session maintains its own independent chat history.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if request.session_id not in sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Call POST /session/start first."
        )

    try:
        agent = sessions[request.session_id]
        result: SQLAgentResponse = agent.ask(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return AskResponse(
        session_id=request.session_id,
        answer=result.answer,
        sql=result.sql,
        confidence=result.confidence,
        category=result.category,
        assumptions=result.assumptions,
        needs_clarification=result.needs_clarification,
        clarification_question=result.clarification_question
    )


@app.delete("/session/{session_id}", tags=["Session"])
def end_session(session_id: str):
    """
    End a session and clear its chat history from memory.
    Call this when the user logs out or closes the chat.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del sessions[session_id]
    return {"status": f"Session {session_id} ended."}


@app.delete("/session/{session_id}/history", tags=["Session"])
def clear_history(session_id: str):
    """Clear chat history for a session without ending it."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    sessions[session_id].chat_history.clear()
    return {"status": f"History cleared for session {session_id}."}