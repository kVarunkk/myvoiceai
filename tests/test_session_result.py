"""SessionResult: the turns and the end reason a session reports back to its caller."""
import asyncio
import unittest
from unittest.mock import patch

from myvoiceai import SessionResult, agent as agent_module, run_voice_session

from test_pipeline_characterization import (
    FakeClientWebSocket,
    FakeLLM,
    PipelineTestCase,
    make_agent,
    text_chunk,
)


class SessionResultTests(PipelineTestCase):
    async def finish(self, timeout=6) -> SessionResult:
        return await asyncio.wait_for(self.run_task, timeout=timeout)

    async def test_turns_equal_the_turn_records_sent_to_the_client(self):
        await self.start_session(FakeLLM([text_chunk("First.")], [text_chunk("Second.")]))
        await self.user_says("one")
        await self.wait_turn_done(1)
        await self.user_says("two")
        await self.wait_turn_done(2)
        self.client.disconnect()

        result = await self.finish()

        self.assertIsInstance(result, SessionResult)
        self.assertEqual(result.turns, self.client.json_of("turn"))
        self.assertEqual([t["user"] for t in result.turns], ["one", "two"])
        self.assertEqual([t["assistant"] for t in result.turns], ["First.", "Second."])

    async def test_client_disconnect(self):
        await self.start_session(FakeLLM())
        self.client.disconnect()

        result = await self.finish()

        self.assertEqual(result.end_reason, "client_disconnect")
        self.assertEqual(result.turns, [])

    async def test_inactivity(self):
        await self.start_session(FakeLLM(), inactivity_timeout_seconds=0.5)

        result = await self.finish()

        self.assertEqual(result.end_reason, "inactivity")

    async def test_time_limit(self):
        await self.start_session(FakeLLM(), max_session_seconds=0.5)

        result = await self.finish()

        self.assertEqual(result.end_reason, "time_limit")

    async def test_disconnect_during_the_goodbye_keeps_the_first_reason(self):
        await self.start_session(FakeLLM(), inactivity_timeout_seconds=0.5)
        self.client.auto_ack = False  # the goodbye never finishes playing
        await asyncio.sleep(1.5)
        self.client.disconnect()

        result = await self.finish(timeout=15)

        self.assertEqual(result.end_reason, "inactivity")

    async def test_error_returns_the_partial_turns_without_raising(self):
        await self.start_session(FakeLLM([text_chunk("First.")]))
        await self.user_says("one")
        await self.wait_turn_done(1)

        async def broken_recv():
            raise RuntimeError("stt crashed")

        self.stt.recv = broken_recv
        self.stt.push({"type": "Metadata"})  # wakes the pending recv; the next one raises

        result = await self.finish()

        self.assertEqual(result.end_reason, "error")
        self.assertEqual([t["user"] for t in result.turns], ["one"])
        self.assertTrue(self.client.closed)

    async def test_times(self):
        await self.start_session(FakeLLM())
        await asyncio.sleep(0.2)
        self.client.disconnect()

        result = await self.finish()

        self.assertLessEqual(result.started_at, result.ended_at)
        self.assertGreaterEqual(result.duration_seconds, 0.2)
        self.assertAlmostEqual(result.duration_seconds, result.ended_at - result.started_at, delta=0.1)


class RunVoiceSessionResultTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_the_agent_result(self):
        client = FakeClientWebSocket()
        expected = SessionResult(turns=[], end_reason="client_disconnect", started_at=1.0, ended_at=2.0, duration_seconds=1.0)

        async def fake_run(self):
            return expected

        with patch.object(agent_module.CustomVoiceAgent, "run", fake_run):
            result = await run_voice_session(client, deepgram_api_key="dg-key")

        self.assertIs(result, expected)

    async def test_setup_failure_is_an_error_result_not_an_exception(self):
        client = FakeClientWebSocket()
        closed_with = []

        async def close(code=1000, reason=None):
            closed_with.append(code)

        client.close = close

        result = await run_voice_session(client, deepgram_api_key=None)

        self.assertEqual(result.end_reason, "error")
        self.assertEqual(result.turns, [])
        self.assertEqual(closed_with, [1011])


class SttPrivacyOptOutTests(unittest.TestCase):
    def test_stt_url_carries_the_opt_out_only_when_set(self):
        default = make_agent(FakeClientWebSocket())._stt_url()
        opted_out = make_agent(FakeClientWebSocket(), stt_privacy_opt_out=True)._stt_url()

        self.assertNotIn("mip_opt_out", default)
        self.assertIn("&mip_opt_out=true", opted_out)
        self.assertEqual(opted_out.replace("&mip_opt_out=true", ""), default)


if __name__ == "__main__":
    unittest.main()
