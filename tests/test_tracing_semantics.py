import asyncio
from importlib.util import find_spec
import os
import unittest
from unittest.mock import patch

from myvoiceai.agent import CustomVoiceAgent
from myvoiceai.tools import get_default_registry


class MockVoiceAgent(CustomVoiceAgent):
    async def _stream_completion(self, messages: list) -> tuple[str, dict[int, dict]]:
        _ = messages
        return "mock response", {}


try:
    HAS_OTEL_SDK = (
        find_spec("opentelemetry.sdk.trace") is not None
        and find_spec("opentelemetry.sdk.trace.export.in_memory_span_exporter") is not None
    )
except ModuleNotFoundError:
    HAS_OTEL_SDK = False


@unittest.skipUnless(HAS_OTEL_SDK, "OpenTelemetry SDK is required")
class LLMTracingSemanticsTests(unittest.TestCase):
    def test_provider_resource_uses_configured_service_and_project_names(self):
        agent = MockVoiceAgent.__new__(MockVoiceAgent)
        tracer = agent._initialize_tracer(
            exporters_config=None,
            service_name="test-service",
            project_name="test-project",
            service_version="test-version",
            deployment_environment="test",
        )
        self.assertIsNotNone(tracer)
        provider = agent._tracer_provider
        self.assertIsNotNone(provider)
        if provider is None:
            self.fail("Tracer provider was not initialized")

        self.assertEqual(provider.resource.attributes["service.name"], "test-service")
        self.assertEqual(
            provider.resource.attributes["openinference.project.name"],
            "test-project",
        )
        self.assertEqual(provider.resource.attributes["service.version"], "test-version")
        self.assertEqual(
            provider.resource.attributes["deployment.environment"],
            "test",
        )
        provider.shutdown()

    def test_version_and_environment_are_not_read_from_environment(self):
        agent = MockVoiceAgent.__new__(MockVoiceAgent)
        with patch.dict(
            os.environ,
            {"APP_VERSION": "env-version", "ENVIRONMENT": "env-environment"},
        ):
            tracer = agent._initialize_tracer(
                exporters_config=None,
                service_name="test-service",
                project_name="test-project",
                service_version=None,
                deployment_environment=None,
            )
        self.assertIsNotNone(tracer)
        provider = agent._tracer_provider
        self.assertIsNotNone(provider)
        if provider is None:
            self.fail("Tracer provider was not initialized")

        self.assertNotIn("service.version", provider.resource.attributes)
        self.assertNotIn("deployment.environment", provider.resource.attributes)
        provider.shutdown()

    def test_session_id_is_generated_when_omitted(self):
        import uuid

        agent = MockVoiceAgent(
            client_websocket=None,
            system_prompt="system",
            max_session_seconds=60,
            inactivity_timeout_seconds=10,
            max_duration_message="bye",
            inactivity_message="inactive",
            greeting_message="hello",
            endpointing=1200,
            utterance_end=2500,
            stable_interim_secs=1.5,
            stable_interim_secs_no_punct=3.0,
            model="mock-model",
            stt_model="mock-stt",
            tts_model="mock-tts",
            tracing=False,
            session_id=None,
            tool_registry=get_default_registry(),
            deepgram_api_key="test-key",
        )
        uuid.UUID(agent.session_id)

    def test_eval_metadata_is_added_to_turn_span_attributes(self):
        agent = MockVoiceAgent(
            client_websocket=None,
            system_prompt="system",
            max_session_seconds=60,
            inactivity_timeout_seconds=10,
            max_duration_message="bye",
            inactivity_message="inactive",
            greeting_message="hello",
            endpointing=1200,
            utterance_end=2500,
            stable_interim_secs=1.5,
            stable_interim_secs_no_punct=3.0,
            model="mock-model",
            stt_model="mock-stt",
            tts_model="mock-tts",
            tracing=False,
            session_id="session-123",
            tool_registry=get_default_registry(),
            deepgram_api_key="test-key",
            eval_run_id="run-1",
            eval_case_id="case-2",
            eval_variant="candidate",
        )
        self.assertEqual(
            agent.eval_attributes,
            {
                "eval.run_id": "run-1",
                "eval.case_id": "case-2",
                "eval.variant": "candidate",
            },
        )

    def test_llm_span_has_openinference_kind_input_and_output(self):
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
            InMemorySpanExporter,
        )

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        tracer = provider.get_tracer("tracing-semantics-test")

        agent = MockVoiceAgent.__new__(MockVoiceAgent)
        agent.interruption_event = asyncio.Event()
        agent._gemini_request_ts = None

        async def invoke_llm():
            with tracer.start_as_current_span(
                "llm_stream",
                attributes={"llm.model": "mock-model"},
            ) as span:
                await agent.llm_call(
                    messages=[{"role": "user", "content": "mock input"}],
                    prompt="mock input",
                    llm_span=span,
                )

        asyncio.run(invoke_llm())
        captured_span = exporter.get_finished_spans()[0]
        provider.shutdown()
        if captured_span is None:
            self.fail("LLM span was not exported")
        attributes = captured_span.attributes
        if attributes is None:
            self.fail("LLM span has no attributes")

        self.assertEqual(attributes["openinference.span.kind"], "LLM")
        self.assertEqual(attributes["input.value"], "mock input")
        self.assertEqual(attributes["output.value"], "mock response")


if __name__ == "__main__":
    unittest.main()
