# Changelog

Notable changes to `myvoiceai` are documented here by release. Versions 0.0.4 and later correspond to repository release tags.

## [0.0.11]

### Added

- `run_voice_session()` returns a `SessionResult` with the session's turns, its `end_reason` (`completed`, `time_limit`, `inactivity`, `client_disconnect` or `error`), start and end times and duration. Errors are returned as `end_reason="error"` with the partial turns instead of being raised. Callers that ignore the return value are unaffected.
- `end_interview_tool`: gives the model an `end_interview` tool. After the closing line has been generated, sent and played, the session closes with code 1000 and `end_reason="completed"`. No filler phrase is spoken for it, and a second call is a no-op.
- `inactivity_prompt_seconds` and `inactivity_prompt_message`: one spoken nudge per silence before the inactivity close. The nudge's own playback doesn't restart the silence timer.
- `trace_content`: with `False`, spans leave out transcripts, replies, tool arguments and results and guardrail matches, and exceptions are recorded by type only.
- `stt_privacy_opt_out`: adds Deepgram's Model Improvement Program opt-out (`mip_opt_out=true`) to the STT connection.
- `ToolRegistry.copy()`.

### Changed

- The session timer no longer acts while a session is already closing.
- The client WebSocket is closed with an explicit code 1000 on cleanup (the previous default).

## [0.0.10]

### Fixed

- `run_voice_session()` no longer fails every session when it receives an unsupported keyword argument, such as the `otel_exporter_endpoint` and `otel_exporter_headers` arguments removed in 0.0.9. Unsupported arguments are logged and ignored, as documented.
- A short reply without sentence or clause punctuation (for example "Sure thing") is no longer followed by the guardrail block message. The block message is now spoken only when no part of the reply was spoken.

### Changed

- Internal refactor of `CustomVoiceAgent` with no change to the public API. OpenTelemetry setup moved to the new `myvoiceai.tracing` module; `OTLPExporterConfig` can still be imported from `myvoiceai.agent`.
- Timing and size values used by the pipeline are now named constants in `myvoiceai.lib.constants`, with unchanged values.

## [0.0.9]

### Breaking changes

- Removed the `otel_exporter_endpoint` and `otel_exporter_headers` arguments from `run_voice_session()` and `CustomVoiceAgent`. Configure every destination with `otel_exporters`, a sequence of dictionaries containing an `endpoint` and optional `headers`.
- Removed the implicit localhost OTLP exporter. With tracing enabled and no exporter configurations, spans are created but are not exported.

### Added

- Support exporting session spans to multiple configured OTLP/HTTP destinations.
- Add OpenInference span kind and input/output attributes to make Arize traces display span types and content.
- Allow configuring the OpenTelemetry service name and OpenInference project name per session.
- Generate a unique session ID when omitted and support deployment and evaluation metadata on traces.
- Require callers to provide service version and deployment environment explicitly; the package no longer reads them from environment variables.

## [0.0.8]

### Fixed

- Fixed OpenTelemetry observability initialization and session cleanup.

## [0.0.7]

### Fixed

- Fixed truncation of goodbye messages.

## [0.0.6]

### Fixed

- Fixed goodbye-message audio cutoff.

## [0.0.5]

### Fixed

- Fixed audio playback in Firefox.

## [0.0.4]

### Added

- Published the package to PyPI.

## [0.0.1]–[0.0.3]

### Added

- Early development releases introduced the voice-agent pipeline, barge-in handling, transcript display and connection controls, LiteLLM support, session timeouts, guardrails, tool calling, observability, and the Gethired integration.
