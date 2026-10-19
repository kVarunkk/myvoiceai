"""inactivity_prompt_seconds: one spoken nudge before the inactivity close."""
import asyncio
import time
import unittest

from myvoiceai.lib.constants import INACTIVITY_PROMPT_MESSAGE

from test_pipeline_characterization import GREETING, FakeLLM, PipelineTestCase, text_chunk, wait_until

BYE = "Bye for now."


class InactivityPromptTests(PipelineTestCase):
    async def test_nudge_fires_once_then_the_session_closes_at_the_timeout(self):
        await self.start_session(FakeLLM(), inactivity_prompt_seconds=0.5, inactivity_timeout_seconds=2.5)
        started = time.monotonic()

        result = await asyncio.wait_for(self.run_task, timeout=10)

        self.assertEqual(
            self.tts.spoken,
            [GREETING, "FLUSH", INACTIVITY_PROMPT_MESSAGE, "FLUSH", "CLEAR", BYE, "FLUSH"],
        )
        self.assertEqual(result.end_reason, "inactivity")
        # The nudge's own playback doesn't restart the silence timer: the close
        # comes on the first one-second tick past 2.5 s of silence, not 2.5 s after the nudge.
        self.assertLess(time.monotonic() - started, 3.9)

    async def test_speech_resets_the_timer_and_allows_another_nudge(self):
        await self.start_session(
            FakeLLM([text_chunk("Ok.")]),
            inactivity_prompt_seconds=0.5,
            inactivity_timeout_seconds=2.5,
        )
        await wait_until(lambda: INACTIVITY_PROMPT_MESSAGE in self.tts.spoken, timeout=3, msg="first nudge")

        await self.user_says("sorry, thinking")
        await self.wait_turn_done(1)
        spoke_at = time.monotonic()
        await wait_until(lambda: BYE in self.tts.spoken, timeout=6, msg="goodbye")

        self.assertGreaterEqual(time.monotonic() - spoke_at, 2.0)
        self.assertEqual(self.tts.spoken.count(INACTIVITY_PROMPT_MESSAGE), 2)
        self.assertLess(
            self.tts.spoken.index("Ok."),
            len(self.tts.spoken) - 1 - self.tts.spoken[::-1].index(INACTIVITY_PROMPT_MESSAGE),
        )
        await asyncio.wait_for(self.run_task, timeout=6)

    async def test_custom_nudge_message(self):
        await self.start_session(
            FakeLLM(),
            inactivity_prompt_seconds=0.5,
            inactivity_prompt_message="Still there?",
            inactivity_timeout_seconds=2.5,
        )
        await wait_until(lambda: "Still there?" in self.tts.spoken, timeout=3, msg="custom nudge")
        await self.end_session()

    async def test_unset_never_nudges(self):
        await self.start_session(FakeLLM(), inactivity_timeout_seconds=1.5)

        await asyncio.wait_for(self.run_task, timeout=6)

        self.assertNotIn(INACTIVITY_PROMPT_MESSAGE, self.tts.spoken)
        self.assertEqual(self.tts.spoken, [GREETING, "FLUSH", "CLEAR", BYE, "FLUSH"])


if __name__ == "__main__":
    unittest.main()
