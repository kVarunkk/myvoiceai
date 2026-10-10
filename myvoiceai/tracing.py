"""Optional OpenTelemetry tracing for voice sessions.

Needs the `observability` extra (`pip install "myvoiceai[observability]"`).
Without it, `trace` and `StatusCode` are None and no spans are created.
"""
import logging
from typing import NotRequired, Sequence, TypedDict

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
except Exception:
    # Tracing extra not installed: the agent's tracer stays None, so no span is ever created.
    trace = None
    StatusCode = None

logger = logging.getLogger("voice_agent")


class OTLPExporterConfig(TypedDict):
    endpoint: str
    headers: NotRequired[dict[str, str]]


def create_tracer_provider(
    exporters_config: Sequence[OTLPExporterConfig] | None,
    service_name: str,
    project_name: str,
    service_version: str | None,
    deployment_environment: str | None,
):
    """Builds a TracerProvider that exports to each configured OTLP/HTTP
    destination. Returns None (and logs a warning) if setup fails."""
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        resource_attributes = {
            "service.name": service_name,
            "openinference.project.name": project_name,
        }
        if service_version:
            resource_attributes["service.version"] = service_version
        if deployment_environment:
            resource_attributes["deployment.environment"] = deployment_environment
        resource = Resource.create(resource_attributes)
        provider = TracerProvider(resource=resource)
        for exporter_index, exporter_config in enumerate(exporters_config or (), start=1):
            exporter_endpoint = exporter_config.get("endpoint")
            if not exporter_endpoint:
                raise ValueError("Each OTLP exporter configuration needs an endpoint.")
            exporter_headers = exporter_config.get("headers", {})
            if not isinstance(exporter_headers, dict):
                raise TypeError("OTLP exporter headers must be a dictionary.")
            logger.info("Initializing OTLP exporter %d.", exporter_index)
            exporter = OTLPSpanExporter(
                endpoint=exporter_endpoint,
                headers=exporter_headers,
            )
            provider.add_span_processor(BatchSpanProcessor(exporter))
        return provider
    except Exception:
        logger.warning(
            "Tracing requested but OpenTelemetry SDK/exporter initialization failed; "
            "install myvoiceai[observability] and check the endpoint configuration.",
            exc_info=True,
        )
        return None


def mark_span_error(span) -> None:
    if StatusCode is not None:
        span.set_status(StatusCode.ERROR)
