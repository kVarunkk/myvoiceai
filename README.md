# myvoiceai

`myvoiceai` is a real-time voice-agent pipeline for FastAPI/WebSocket applications. It streams microphone audio to Deepgram STT, sends recognized turns to an LLM through LiteLLM, streams speech through Deepgram TTS, and returns PCM audio and transcript/control messages to the WebSocket client.

## Installation

Install the package and its runtime dependencies:

```bash
pip install myvoiceai
```

Install optional OpenTelemetry tracing support when you need OTLP export:

```bash
pip install "myvoiceai[observability]"
```

The package requires Python 3.11 or newer. Set provider credentials as secrets in your environment or hosting platform; do not commit them to source control.

## Quickstart

Call `run_voice_session()` from an accepted FastAPI WebSocket endpoint. The default tool registry is used if `tool_registry` is omitted.

```python
from fastapi import FastAPI, WebSocket
from myvoiceai import run_voice_session

app = FastAPI()

@app.websocket("/ws/voice")
async def voice_session(websocket: WebSocket):
    await websocket.accept()
    await run_voice_session(
        websocket=websocket,
        system_prompt="You are a concise travel assistant.",
        greeting_message="Hello! How can I help you today?",
        llm_provider_api_key="your-llm-provider-key",
        deepgram_api_key="your-deepgram-key",
        session_id="session-001",
    )
```

For deployment, read credentials from environment variables or a secret manager instead of hardcoding them. The LLM model defaults to `gemini/gemini-2.5-flash`; provide the corresponding provider key or choose a different LiteLLM model and key.

## FastAPI Example

The repository includes a runnable example in `example/fastapi_app.py`, with its browser client in `example/static/index.html`. From the repository root, install the dependencies and run:

```bash
pip install -e ".[observability]"
uvicorn example.fastapi_app:app --host 0.0.0.0 --port 8000
```

The example reads `GEMINI_API_KEY` and `DEEPGRAM_API_KEY` from environment variables (and loads a local `.env` file if present). Its `__main__` block uses port `8002`; the Uvicorn command above explicitly uses port `8000`. The `example/` directory is part of the source repository and is not included as a Python package by the current setuptools package configuration.

## Session API

`run_voice_session(websocket, ...)` accepts these options:

| Parameter | Purpose | Default |
| --- | --- | --- |
| `system_prompt` | Instructions prepended to the conversation | Concise voice-assistant prompt |
| `greeting_message` | Text spoken at the start of a session | `Hi there! How can I help you today?` |
| `inactivity_message` | Spoken message when inactivity ends the session | Built-in inactivity message |
| `max_duration_message` | Spoken message when the session duration limit is reached | Built-in time-limit message |
| `max_session_seconds` | Maximum session duration in seconds | `300` |
| `inactivity_timeout_seconds` | Inactivity threshold in seconds | `10` |
| `model` | LiteLLM model identifier | `gemini/gemini-2.5-flash` |
| `llm_provider_api_key` | API key for the selected LLM provider | `None` (supply a valid key) |
| `deepgram_api_key` | Deepgram STT/TTS API key | `None` (required) |
| `stt_model` | Deepgram speech-recognition model | `nova-2` |
| `tts_model` | Deepgram text-to-speech voice model | `aura-asteria-en` |
| `endpointing` | Deepgram endpointing interval in milliseconds | `1200` |
| `utterance_end` | Deepgram utterance-end interval in milliseconds | `2500` |
| `stable_interim_secs` | Delay before dispatching a stable punctuated interim transcript | `1.5` |
| `stable_interim_secs_no_punct` | Delay for a stable interim without punctuation | `3.0` |
| `tool_registry` | Registry of callable tools; omit to use the package default registry | Package default registry |
| `tracing` | Enable OpenTelemetry spans for the session | `False` |
| `otel_exporters` | List of OTLP/HTTP exporter configurations (`endpoint` and optional `headers`) | `None` |
| `service_name` | OpenTelemetry `service.name` resource attribute | `myvoiceai` |
| `project_name` | OpenInference `openinference.project.name` resource attribute | `myvoiceai-voice-session` |
| `service_version` | OpenTelemetry `service.version` resource attribute | `None` |
| `deployment_environment` | Deployment environment resource attribute | `None` |
| `session_id` | Identifier attached to session traces/logging; a UUID is generated when omitted | Generated UUID |
| `eval_run_id` | Optional evaluation-run identifier attached to each turn span | `None` |
| `eval_case_id` | Optional evaluation-case identifier attached to each turn span | `None` |
| `eval_variant` | Optional prompt/model variant label attached to each turn span | `None` |

`max_duration_message` and `inactivity_message` are also accepted by `run_voice_session()`. Unsupported extra keyword arguments are logged and ignored.

`CustomVoiceAgent` is the lower-level class used by `run_voice_session()`. Prefer the function for normal application integration; direct class construction requires the full constructor argument set.

## Custom Tools

Create a `ToolRegistry`, register each tool with an OpenAI-style function schema and an async implementation, and pass it to `run_voice_session()`:

```python
from myvoiceai.tools import ToolRegistry

registry = ToolRegistry()

async def lookup_city(city: str):
    return {"city": city, "country": "example"}

registry.register(
    name="lookup_city",
    schema={
        "type": "function",
        "function": {
            "name": "lookup_city",
            "description": "Look up a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    impl=lookup_city,
)

await run_voice_session(
    websocket=websocket,
    llm_provider_api_key="your-llm-provider-key",
    deepgram_api_key="your-deepgram-key",
    tool_registry=registry,
)
```

Tool implementations may be async callables. Tool schemas are supplied to LiteLLM and tool results are added to the conversation. Avoid putting secrets or sensitive personal information in tool results if traces or application logs may contain them.

## OpenTelemetry Tracing

Tracing is optional. Install the extra and pass `tracing=True`. Exporter endpoints and headers are passed to the session API, so configure them from trusted application settings:

```python
import os

await run_voice_session(
    websocket=websocket,
    llm_provider_api_key=os.environ["LLM_PROVIDER_API_KEY"],
    deepgram_api_key=os.environ["DEEPGRAM_API_KEY"],
    tracing=True,
    service_name="travel-assistant",
    project_name="travel-assistant-evals",
    otel_exporters=[
        {
            "endpoint": os.environ["OTEL_EXPORTER_ENDPOINT"],
            "headers": {
                "x-honeycomb-team": os.environ["OTEL_EXPORTER_API_KEY"],
            },
        },
    ],
)
```

Use the complete OTLP/HTTP traces URL supplied by your backend, including its path (commonly `/v1/traces`). The package creates a provider for each traced session and shuts it down during session cleanup to flush spans. If `otel_exporters` is omitted or empty, spans are created but no exporter is registered. If the optional SDK/exporter cannot be initialized, tracing is disabled for that agent and a warning is logged.

To send each session's spans to multiple OTLP/HTTP destinations, add one configuration dictionary per destination:

```python
await run_voice_session(
    websocket=websocket,
    tracing=True,
    otel_exporters=[
        {
            "endpoint": os.environ["HONEYCOMB_OTEL_ENDPOINT"],
            "headers": {
                "x-honeycomb-team": os.environ["HONEYCOMB_API_KEY"],
            },
        },
        {
            "endpoint": os.environ["ARIZE_OTEL_ENDPOINT"],
            "headers": {
                "authorization": os.environ["ARIZE_API_KEY"],
                "arize-space-id": os.environ["ARIZE_SPACE_ID"],
                "arize-interface": "otel",
            },
        },
    ],
)
```

Each configuration must include an `endpoint` and may include `headers`. The package creates and manages the OTLP exporters internally, so application code does not need to import OpenTelemetry exporters. The `myvoiceai[observability]` extra is still required in the environment where tracing runs.

Agent spans include OpenInference semantic attributes (`openinference.span.kind`, `input.value`, and `output.value` where applicable) so Arize can classify span types and display inputs/outputs. These attributes may contain conversation or tool data; review the data before exporting it to a tracing backend.

`service_name` and `project_name` are resource attributes shared by all spans from a session. Set them to stable application and project identifiers to group traces consistently across sessions. `project_name` is distinct from Arize's Space ID, which remains an exporter header.

`service_version` and `deployment_environment` add release and environment context to the resource. The package does not read these values from environment variables; pass them explicitly when calling `run_voice_session()`. If omitted, the corresponding resource attribute is omitted. For evaluation runs, pass `eval_run_id`, `eval_case_id`, and `eval_variant`; these are attached to each `voice_turn` span for filtering and comparison.

### Breaking change in 0.0.9

Version 0.0.9 removes the `otel_exporter_endpoint` and `otel_exporter_headers` arguments. Replace them with `otel_exporters`, a list of dictionaries with an `endpoint` and optional `headers`. For example, replace:

```python
otel_exporter_endpoint=endpoint,
otel_exporter_headers=headers,
```

with:

```python
otel_exporters=[{"endpoint": endpoint, "headers": headers}],
```

There is no implicit localhost exporter in 0.0.9. If no configurations are supplied, no spans are exported.

The repository contains local observability examples under `example/`:

- `example/docker-compose.yml` starts Jaeger, an OpenTelemetry Collector, Prometheus, and Grafana.
- `example/otel-collector-config.yaml` accepts OTLP/HTTP on port `4318` and forwards traces to Jaeger over OTLP/gRPC.
- Jaeger UI is available at `http://localhost:16686` when the example stack is running.

For production, you can send directly to a hosted OTLP-compatible backend or use a private Collector as a relay. Keep exporter credentials in deployment secrets, use TLS for network connections, and do not accept arbitrary exporter endpoints from untrusted clients. Review trace attributes and retention requirements: this package records conversation and tool data in some spans, which may include sensitive information.

## WebSocket Audio and Messages

The server expects incoming binary audio as mono, 16-bit linear PCM at 16 kHz, matching the configured Deepgram STT stream. The example browser captures and sends this format. The server returns binary mono, 16-bit linear PCM audio at 16 kHz for playback.

The client should send this JSON control message after an utterance has finished playing:

```json
{"control":"playback_complete"}
```

The server sends these control messages:

- `{"control":"utterance_end"}` marks the end of the current generated audio utterance. Wait for scheduled audio to finish before sending `playback_complete`.
- `{"control":"clear_speaker_buffer"}` tells the client to stop and discard queued playback, typically after barge-in.

The server may also send JSON transcript messages:

- `transcript_chunk`: partial transcript with `role`, `turn_id`, and `text`; user chunks can include `replace: true`.
- `transcript`: finalized transcript with `role`, `text`, and `turn_id`.
- `turn`: completed turn containing `turn_id`, `timestamp`, `user`, `assistant`, and `interrupted`.

The client must handle both binary audio frames and JSON text frames on the same WebSocket.

## Package Exports

The top-level `myvoiceai` package exports `run_voice_session`, `CustomVoiceAgent`, `ToolRegistry`, and `get_default_registry`.
