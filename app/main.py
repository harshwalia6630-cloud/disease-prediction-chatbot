"""FastAPI service: REST endpoints plus a small web chat UI.

Run locally:  uvicorn app.main:app --reload
"""

import sys
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from symptom_chatbot.chatbot import DISCLAIMER, DiseaseChatbot, Session  # noqa: E402

MAX_SESSIONS = 5_000
SESSION_TTL_S = 30 * 60

app = FastAPI(title="Disease Prediction Chatbot", version="1.0.0",
              description="Symptom-based disease prediction with TF-IDF + LinearSVC. " + DISCLAIMER)
bot = DiseaseChatbot()
sessions: "OrderedDict[str, tuple[Session, float]]" = OrderedDict()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, examples=["I have a high fever, chills and body aches"])
    top_k: int = Field(3, ge=1, le=10)


class ChatRequest(BaseModel):
    message: str = Field(..., max_length=2000)
    session_id: str | None = None


def _get_session(session_id):
    now = time.time()
    while sessions and next(iter(sessions.values()))[1] < now - SESSION_TTL_S:
        sessions.popitem(last=False)
    if session_id in sessions:
        session = sessions.pop(session_id)[0]
    else:
        session_id, session = uuid.uuid4().hex, Session()
    sessions[session_id] = (session, now)
    if len(sessions) > MAX_SESSIONS:
        sessions.popitem(last=False)
    return session_id, session


@app.get("/health")
def health():
    return {"status": "ok", "diseases": len(bot.classes)}


@app.post("/api/predict")
def predict(req: PredictRequest):
    t0 = time.perf_counter()
    preds = bot.predict(req.text, req.top_k)
    return {
        "predictions": preds,
        "symptoms_detected": bot.detect_symptoms(req.text),
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "disclaimer": DISCLAIMER,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    session_id, session = _get_session(req.session_id)
    return {"session_id": session_id, **bot.respond(session, req.message)}


@app.delete("/api/chat/{session_id}")
def end_chat(session_id: str):
    if sessions.pop(session_id, None) is None:
        raise HTTPException(404, "unknown session")
    return {"ended": session_id}


app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")
