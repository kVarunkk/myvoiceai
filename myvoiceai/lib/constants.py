OTEL_EXPORTER_ENDPOINT="http://localhost:4318/v1/traces"
DEEPGRAM_STT_MODEL="nova-2"
DEEPGRAM_TTS_MODEL="aura-asteria-en"
DEEPGRAM_STT_URL = (
    "wss://api.deepgram.com/v1/listen"
    "?model={stt_model}&encoding=linear16&sample_rate=16000&channels=1"
    "&interim_results=true&endpointing={endpointing}&smart_format=true"
    "&no_delay=true"
    "&vad_events=true"
    "&utterance_end_ms={utterance_end}"
)

# Deepgram's Model Improvement Program opt-out: the audio isn't kept for training.
DEEPGRAM_MIP_OPT_OUT_PARAM = "&mip_opt_out=true"

DEEPGRAM_TTS_URL = (
    "wss://api.deepgram.com/v1/speak"
    "?encoding=linear16&sample_rate=16000&model={tts_model}"
)
LLM_MODEL = "gemini/gemini-2.5-flash"
LLM_REQUEST_TIMEOUT_SECONDS = 30.0

SENTENCE_BOUNDARY_CHARS = {".", "!", "?", "\n"}
CLAUSE_BOUNDARY_CHARS = {",", ";", ":"}
MAX_BUFFER_CHARS_BEFORE_FORCED_FLUSH = 100
MIN_CHARS_BEFORE_CLAUSE_FLUSH = 40  # a clause boundary only ends a TTS chunk past this many chars

DEFAULT_MAX_SESSION_SECONDS = 300          
DEFAULT_INACTIVITY_TIMEOUT_SECONDS = 10    
DEFAULT_GOODBYE_WAIT_SECONDS = 10 
GOODBYE_TTS_GRACE_SECONDS = 0.3  # let TTS start on the goodbye before waiting on it

DEFAULT_GREETING_MESSAGE = "Hi there! How can I help you today?"

SYSTEM_PROMPT = (
    "You are a helpful, concise voice assistant. Keep replies short and "
    "conversational since they will be spoken aloud."
)

BASE_SYSTEM_PROMPT = "Do not use markdown formatting (no asterisks, bullet points, headers, or bold/italic syntax). Your responses are converted to speech, so write in plain spoken sentences only."

GUARDRAIL_BLOCK_MESSAGE = "I can't help with that."
MAX_DURATION_MESSAGE="We've reached our time limit for this session, goodbye for now."
INACTIVITY_MESSAGE="I haven't heard from you in a bit, so I'll go ahead and close this session."
TOOL_FILLER_PHRASES = [
    "Let me check on that.",
    "One sec, looking into it.",
    "Give me a moment.",
]
TOOL_CALL_TIMEOUT_SECONDS = 8.0
MAX_TOOL_HOPS = 3

DEFAULT_ENDPOINTING = 1200
DEFAULT_UTTERANCE_END = 2500
STABLE_INTERIM_SECS = 1.5
STABLE_INTERIM_NO_PUNCT_SECS = 3.0
STABLE_INTERIM_POLL_SECS = 0.05
STABLE_DISPATCH_DEDUP_SECS = 5  # window for dropping words already sent via a stable interim

FILLERS = {"and", "uh", "um", "so", "but", "or", "the", "a", "like"}

TRACE_TEXT_MAX_CHARS = 500  # longest text recorded on a span attribute
INACTIVITY_PROMPT_MESSAGE = "Take your time; let me know when you're ready."

END_INTERVIEW_TOOL_NAME = "end_interview"
END_INTERVIEW_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": END_INTERVIEW_TOOL_NAME,
        "description": (
            "Ends the interview and closes the session. Call it only after you "
            "have spoken your closing line; say nothing after calling it."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}
# Spoken when the model ends the interview without a closing line of its own.
END_INTERVIEW_FALLBACK_MESSAGE = "Thank you for your time. Goodbye."
