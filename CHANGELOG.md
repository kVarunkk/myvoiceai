# Changelog

Notable changes to `myvoiceai` are documented here by release. Versions 0.0.4 and later correspond to repository release tags.

## [0.0.9] - Unreleased

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
