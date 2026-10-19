"""
FastAPI example for myvoiceai.
This demonstrates how to use `run_voice_session` with a WebSocket endpoint,
and how to read the `SessionResult` it returns.

Run from the repository root:  python -m example.fastapi_app
Then open http://localhost:8002 and pick a scenario and options.

The WebSocket takes these query parameters (all optional):
- scenario:        "interview" (default) uses end_interview_tool; "travel" uses the lookup_flight tool only
- trace_content:   "1" (default) or "0" to leave conversation text out of spans
- privacy:         "1" to add Deepgram's mip_opt_out=true to the STT connection
- nudge_seconds:   seconds of silence before the spoken nudge (default 8, "0" turns it off)
- idle_seconds:    seconds of silence before the inactivity close (default 20)
- max_seconds:     session time limit (default: the package default)
"""
import logging
import os
import uuid
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from myvoiceai import SessionResult, run_voice_session
from myvoiceai.lib.constants import DEFAULT_MAX_SESSION_SECONDS
from myvoiceai.tools import ToolRegistry
from myvoiceai.tracing import OTLPExporterConfig

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("example")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")

app = FastAPI(title="Voice AI Example")

# Allow all CORS for browser clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent


# One registry shared by every session. end_interview_tool=True registers its
# tool on a copy, so this registry must still list only lookup_flight afterwards.
async def lookup_flight(**kwargs):
    # This is a placeholder - replace with real logic
    return {
        "flight": "AA123",
        "gate": "B12",
        "departure": kwargs.get("from"),
        "destination": kwargs.get("to"),
    }


shared_registry = ToolRegistry()
shared_registry.register(
    name="lookup_flight",
    schema={
        "type": "function",
        "function": {
            "name": "lookup_flight",
            "description": "Look up a flight",
            "parameters": {
                "type": "object",
                "properties": {
                    "from": {"type": "string", "description": "Departure airport code"},
                    "to": {"type": "string", "description": "Destination airport code"},
                },
                "required": ["from", "to"],
            },
        },
    },
    impl=lookup_flight,
)

SCENARIOS = {
    "interview": {
        "system_prompt": (
            "You are a friendly screening interviewer for a Junior Python Developer role. "
            "Ask these questions one at a time, waiting for each answer:\n"
            "1. Tell me about a Python project you have worked on.\n"
            "2. How do you debug a function that returns the wrong result?\n"
            "Keep each reply to one or two short sentences. "
            "After the candidate answers the last question, or if they ask to stop, "
            "say one short closing line thanking them, then call the end_interview tool. "
            "Say nothing after calling it."
        ),
        "greeting_message": "Hi, Thanks for joining this short screening interview. Are you ready to start?",
        "end_interview_tool": True,
    },
    "travel": {
        "system_prompt": "You are a travel assistant. You can look up flight information using the lookup_flight tool.",
        "greeting_message": "Hi there! I'm your travel assistant. ",
        "end_interview_tool": False,
    },
}

# Finished sessions by id, so the page can show each SessionResult.
session_results: dict[str, SessionResult] = {}
# Sessions whose run_voice_session hasn't returned yet. The client socket is
# closed before the tracer flushes its spans, so a session stays here for a
# few seconds after the page sees the close.
running_sessions: set[str] = set()


def otel_exporters() -> list[OTLPExporterConfig]:
    """The exporters whose settings are in the environment."""
    exporters: list[OTLPExporterConfig] = []
    if os.getenv("OTEL_EXPORTER_ENDPOINT"):
        exporters.append({
            "endpoint": os.environ["OTEL_EXPORTER_ENDPOINT"],
            "headers": (
                {"x-honeycomb-team": os.environ["OTEL_EXPORTER_API_KEY"]}
                if os.getenv("OTEL_EXPORTER_API_KEY")
                else {}
            ),
        })
    if os.getenv("ARIZE_API_KEY") and os.getenv("ARIZE_SPACE_ID"):
        exporters.append({
            "endpoint": "https://otlp.arize.com/v1/traces",
            "headers": {
                "arize-api-key": os.environ["ARIZE_API_KEY"],
                "arize-space-id": os.environ["ARIZE_SPACE_ID"],
            },
        })
    return exporters


@app.get("/")
async def root():
    html_path = BASE_DIR / "static" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read(), status_code=200)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id in running_sessions:
        return JSONResponse({"status": "finishing"}, status_code=202)
    result = session_results.get(session_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown session")
    return asdict(result)


@app.get("/registry")
async def get_registry():
    """The shared registry's tools: should stay ["lookup_flight"] after interview sessions."""
    return {"tools": [schema["function"]["name"] for schema in shared_registry.schemas()]}


@app.websocket("/ws/voice")
async def voice_session(websocket: WebSocket):
    await websocket.accept()

    params = websocket.query_params
    scenario = SCENARIOS.get(params.get("scenario", "interview"), SCENARIOS["interview"])
    nudge_seconds = float(params.get("nudge_seconds", "8"))
    max_seconds = params.get("max_seconds")
    exporters = otel_exporters()

    # Tell the page its session id, so it can fetch the result after the close.
    session_id = f"example-{uuid.uuid4().hex[:8]}"
    await websocket.send_json({"session": {"id": session_id}})

    running_sessions.add(session_id)
    try:
        result = await run_voice_session(
            websocket=websocket,
            system_prompt=scenario["system_prompt"],
            greeting_message=scenario["greeting_message"],
            endpointing=1200,
            utterance_end=2500,
            stable_interim_secs=1.5,
            stable_interim_secs_no_punct=3.0,
            inactivity_prompt_seconds=nudge_seconds or None,
            inactivity_timeout_seconds=int(params.get("idle_seconds", "20")),
            max_session_seconds=float(max_seconds) if max_seconds else DEFAULT_MAX_SESSION_SECONDS,
            end_interview_tool=scenario["end_interview_tool"],
            trace_content=params.get("trace_content", "1") != "0",
            stt_privacy_opt_out=params.get("privacy") == "1",
            tool_registry=shared_registry,
            llm_provider_api_key=GEMINI_API_KEY,
            deepgram_api_key=DEEPGRAM_API_KEY,
            session_id=session_id,
            tracing=bool(exporters),
            otel_exporters=exporters or None,
        )
    finally:
        running_sessions.discard(session_id)

    session_results[session_id] = result
    logger.info(
        "Session %s ended: end_reason=%s, %d turns, %.1fs",
        session_id, result.end_reason, len(result.turns), result.duration_seconds,
    )


if __name__ == "__main__":
    import uvicorn

    print("Starting Voice AI server...")
    print("Open http://localhost:8002 to talk to the agent")
    print("Connect WebSocket clients to ws://localhost:8002/ws/voice")

    uvicorn.run("example.fastapi_app:app", host="0.0.0.0", port=8002, reload=True)
