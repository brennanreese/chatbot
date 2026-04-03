"""
FastAPI server for the Bookly customer support chatbot.
Serves the web UI and handles chat API requests.
"""

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from agent import BooklyAgent

# --- Logging & config ---

logging.basicConfig(level=logging.INFO)
logging.getLogger("bookly").setLevel(logging.DEBUG)

load_dotenv()

# --- Session management ---
# Each session ID maps to a BooklyAgent with its own conversation history.
# Stored in memory — lost on restart. See DESIGN.md for production alternatives.

sessions: dict[str, BooklyAgent] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Clean up sessions when the server shuts down."""
    yield
    sessions.clear()


app = FastAPI(title="Bookly Support Agent", lifespan=lifespan)


# --- Middleware ---
# Adds X-Process-Time header to every response for latency visibility.

@app.middleware("http")
async def add_timing(request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time"] = f"{time.time() - start:.3f}s"
    return response


# --- Request / response models ---

class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class ResetRequest(BaseModel):
    session_id: str | None = None


class RatingRequest(BaseModel):
    session_id: str
    rating: int  # 1-5 stars


# --- API routes ---

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Main chat endpoint. Creates a new session if none exists, then delegates to the agent."""
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_API_KEY environment variable not set")
        model = os.environ.get("GEMINI_MODEL")
        sessions[session_id] = BooklyAgent(api_key=api_key, model=model) if model else BooklyAgent(api_key=api_key)

    agent = sessions[session_id]

    try:
        reply = agent.chat(req.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    return ChatResponse(session_id=session_id, reply=reply)


@app.post("/api/reset")
async def reset_session(req: ResetRequest):
    """Clear conversation history for a session so the customer can start fresh."""
    session_id = req.session_id
    if session_id and session_id in sessions:
        sessions[session_id].reset()
    return {"status": "ok"}


@app.get("/api/events/{session_id}")
async def get_events(session_id: str):
    """Return the event trace for a session. Powers the debug panel in the UI."""
    if session_id not in sessions:
        return {"events": []}
    return {"events": sessions[session_id].events}


@app.post("/api/rating")
async def rate_session(req: RatingRequest):
    """Record a 1-5 star rating for a session. Stored in the event trace."""
    if req.rating < 1 or req.rating > 5:
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    agent = sessions[req.session_id]
    agent._log_event("rating", stars=req.rating)
    return {"status": "ok"}


@app.get("/api/stats")
async def get_stats():
    """Compute aggregate metrics across all active sessions."""
    total = len(sessions)
    if total == 0:
        return {"sessions": 0}

    ratings = []
    deflected = 0
    resolved = 0
    action_sessions = 0
    total_turns = 0

    action_tools = {"initiate_return", "send_password_reset"}

    for agent in sessions.values():
        events = agent.events
        roles = [e["role"] for e in events]

        # Collect ratings
        for e in events:
            if e["role"] == "rating":
                ratings.append(e["stars"])

        # Count customer turns
        user_turns = roles.count("user")
        total_turns += user_turns

        # Deflection: no human escalation occurred (API errors don't count)
        has_escalation = any(
            (e["role"] == "escalation" and e.get("reason") != "api_failure") or
            (e["role"] == "tool_call" and e.get("tool") == "escalate_to_human")
            for e in events
        )
        if not has_escalation and user_turns > 0:
            deflected += 1

        # Resolution: only measured for sessions where an action was attempted
        has_action = any(
            e["role"] == "tool_call" and e.get("tool") in action_tools
            for e in events
        )
        if has_action:
            action_sessions += 1
            if not has_escalation:
                resolved += 1

    return {
        "sessions": total,
        "avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
        "total_ratings": len(ratings),
        "deflection_rate": round(deflected / total, 2) if total else 0,
        "resolution_rate": round(resolved / action_sessions, 2) if action_sessions else None,
        "avg_turns": round(total_turns / total, 1) if total else 0,
    }


# --- Static file serving ---

@app.get("/")
async def index():
    """Serve the chat UI."""
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
