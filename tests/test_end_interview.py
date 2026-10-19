"""end_interview: the model ends the session itself after its closing line."""
import asyncio
import json
import unittest

from myvoiceai.lib.constants import (
    END_INTERVIEW_FALLBACK_MESSAGE,
    END_INTERVIEW_TOOL_NAME,
    GUARDRAIL_BLOCK_MESSAGE,
    TOOL_FILLER_PHRASES,
)
from myvoiceai.tools import ToolRegistry

from test_pipeline_characterization import (
    FLUSH,
    GREETING,
    FakeClientWebSocket,
    FakeLLM,
    PipelineTestCase,
    make_agent,
    text_chunk,
    tool_call_chunk,
    wait_until,
)

CLOSING = "Thanks for your time, goodbye."


def end_call(call_id="end-1"):
    return tool_call_chunk(0, call_id, END_INTERVIEW_TOOL_NAME, "{}")


class EndInterviewTests(PipelineTestCase):
    async def start(self, *scripts, **kwargs):
        await self.start_session(FakeLLM(*scripts), end_interview_tool=True, **kwargs)
        self.closed_with = []
        original_close = self.client.close

        async def close(code=1000, reason=None):
            self.closed_with.append(code)
            await original_close(code, reason)

        self.client.close = close

    def hold_tts_flushes(self):
        """Makes the TTS fake keep its answers to Flush until release() is called."""
        held = []

        async def send(data):
            if isinstance(data, str):
                data = json.loads(data)
            self.tts.sent.append(data)
            if data == FLUSH:
                held.extend([b"pcm", json.dumps({"type": "Flushed"})])

        def release():
            for item in held:
                self.tts.incoming.put_nowait(item)
            held.clear()

        self.tts.send = send
        return release

    async def test_closing_line_then_close_with_1000(self):
        await self.start([text_chunk(CLOSING), end_call()])

        await self.user_says("that's all from me")
        result = await asyncio.wait_for(self.run_task, timeout=5)

        self.assertEqual(result.end_reason, "completed")
        self.assertEqual(self.tts.spoken, [GREETING, "FLUSH", CLOSING, "FLUSH", "FLUSH"])
        self.assertNotIn(TOOL_FILLER_PHRASES[0], self.tts.spoken)
        self.assertEqual(len(self.llm.calls), 1)  # the model isn't asked again after ending
        self.assertEqual([(t["user"], t["assistant"]) for t in result.turns], [("that's all from me", CLOSING)])
        self.assertEqual(self.closed_with, [1000])

    async def test_waits_for_the_closing_audio_and_its_playback(self):
        await self.start([text_chunk(CLOSING), end_call()])
        release = self.hold_tts_flushes()
        self.client.auto_ack = False

        await self.user_says("bye")
        await wait_until(lambda: CLOSING in self.tts.spoken, msg="closing line sent to TTS")
        await asyncio.sleep(0.3)
        self.assertFalse(self.client.closed, "closed before Deepgram finished the closing audio")

        release()
        await wait_until(lambda: self.client.sent_bytes.count(b"pcm") >= 2, msg="closing audio sent")
        await asyncio.sleep(0.3)
        self.assertFalse(self.client.closed, "closed before the client played the closing audio")

        self.client.send_text({"control": "playback_complete"})
        result = await asyncio.wait_for(self.run_task, timeout=5)
        self.assertEqual(result.end_reason, "completed")
        self.assertTrue(self.client.closed)

    async def test_a_stale_flush_count_from_a_barge_in_is_corrected_by_cleared(self):
        await self.start([text_chunk(CLOSING), end_call()])
        # A barge-in sent Clear with one Flush unanswered; Deepgram confirms with Cleared.
        self.agent._tts_flushes_sent += 1
        self.agent._flushes_sent_at_clear = self.agent._tts_flushes_sent
        self.tts.push({"type": "Cleared"})
        await wait_until(
            lambda: self.agent._tts_flushes_received == self.agent._tts_flushes_sent,
            msg="Cleared applied",
        )

        await self.user_says("bye")
        result = await asyncio.wait_for(self.run_task, timeout=5)

        self.assertEqual(result.end_reason, "completed")

    async def test_second_call_is_a_no_op(self):
        both = text_chunk(tool_calls=[
            tool_call_chunk(0, "end-1", END_INTERVIEW_TOOL_NAME, "{}").choices[0].delta.tool_calls[0],
            tool_call_chunk(1, "end-2", END_INTERVIEW_TOOL_NAME, "{}").choices[0].delta.tool_calls[0],
        ], finish_reason="tool_calls")
        await self.start([text_chunk(CLOSING), both])

        await self.user_says("bye")
        result = await asyncio.wait_for(self.run_task, timeout=5)

        tool_results = [json.loads(m["content"]) for m in self.agent.conversation_history if m["role"] == "tool"]
        self.assertEqual(tool_results, [{"ok": True}, {"ok": True, "already_ending": True}])
        self.assertEqual(result.end_reason, "completed")
        self.assertEqual(self.tts.spoken.count(CLOSING), 1)
        self.assertEqual(self.closed_with, [1000])

    async def test_no_closing_text_speaks_the_fallback_not_the_block_message(self):
        await self.start([end_call()])

        await self.user_says("bye")
        result = await asyncio.wait_for(self.run_task, timeout=5)

        self.assertEqual(result.end_reason, "completed")
        self.assertIn(END_INTERVIEW_FALLBACK_MESSAGE, self.tts.spoken)
        self.assertNotIn(GUARDRAIL_BLOCK_MESSAGE, self.tts.spoken)

    async def test_the_session_timer_does_not_interrupt_the_closing(self):
        await self.start([text_chunk(CLOSING), end_call()], max_session_seconds=1.5)
        release = self.hold_tts_flushes()

        await self.user_says("bye")
        await wait_until(lambda: CLOSING in self.tts.spoken, msg="closing line sent to TTS")
        await asyncio.sleep(2.2)  # past the time limit
        release()
        result = await asyncio.wait_for(self.run_task, timeout=5)

        self.assertEqual(result.end_reason, "completed")
        self.assertNotIn("Time is up.", self.tts.spoken)


class EndInterviewRegistryTests(unittest.TestCase):
    def test_off_by_default(self):
        agent = make_agent(FakeClientWebSocket())

        self.assertFalse(agent.tool_registry.has(END_INTERVIEW_TOOL_NAME))

    def test_on_adds_the_tool_without_changing_the_callers_registry(self):
        registry = ToolRegistry()

        async def ping():
            return {}

        registry.register(name="ping", schema={"type": "function", "function": {"name": "ping"}}, impl=ping)
        agent = make_agent(FakeClientWebSocket(), tool_registry=registry, end_interview_tool=True)

        self.assertTrue(agent.tool_registry.has("ping"))
        self.assertTrue(agent.tool_registry.has(END_INTERVIEW_TOOL_NAME))
        self.assertEqual(
            [s["function"]["name"] for s in agent.tool_registry.schemas()],
            ["ping", END_INTERVIEW_TOOL_NAME],
        )
        self.assertFalse(registry.has(END_INTERVIEW_TOOL_NAME))


if __name__ == "__main__":
    unittest.main()
