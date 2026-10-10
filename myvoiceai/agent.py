import asyncio
import contextlib
import json
import logging
import random
import re
import time
import uuid
from typing import NotRequired, Sequence, TypedDict, cast

import websockets
from litellm import CustomStreamWrapper, acompletion, exceptions
from starlette.websockets import WebSocketDisconnect
from websockets.exceptions import ConnectionClosed

from myvoiceai.lib.constants import (
    BASE_SYSTEM_PROMPT,
    CLAUSE_BOUNDARY_CHARS,
    DEEPGRAM_STT_MODEL,
    DEEPGRAM_STT_URL,
    DEEPGRAM_TTS_MODEL,
    DEEPGRAM_TTS_URL,
    DEFAULT_ENDPOINTING,
    DEFAULT_GOODBYE_WAIT_SECONDS,
    DEFAULT_GREETING_MESSAGE,
    DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
    DEFAULT_MAX_SESSION_SECONDS,
    DEFAULT_UTTERANCE_END,
    FILLERS,
    GUARDRAIL_BLOCK_MESSAGE,
    INACTIVITY_MESSAGE,
    LLM_MODEL,
    MAX_BUFFER_CHARS_BEFORE_FORCED_FLUSH,
    MAX_DURATION_MESSAGE,
    MAX_TOOL_HOPS,
    SENTENCE_BOUNDARY_CHARS,
    STABLE_INTERIM_NO_PUNCT_SECS,
    STABLE_INTERIM_SECS,
    SYSTEM_PROMPT,
    TOOL_CALL_TIMEOUT_SECONDS,
    TOOL_FILLER_PHRASES,
)
from myvoiceai.tools import ToolRegistry, get_default_registry
from myvoiceai.utils.guardrails import check_input, check_output
from myvoiceai.utils.strip_markdown import strip_markdown_for_speech


class OTLPExporterConfig(TypedDict):
    endpoint: str
    headers: NotRequired[dict[str, str]]

try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode
except Exception:
    # Tracing extra not installed: self.tracer stays None, so no span is ever created.
    trace = None
    StatusCode = None

logger = logging.getLogger("voice_agent")
_AUDIO_UTTERANCE_END = object()

async def _connect_deepgram(url: str, api_key: str | None):
    """Connect to a Deepgram websocket, tolerating both old and new
    versions of the `websockets` library (the auth-header kwarg was
    renamed from extra_headers to additional_headers in v14)."""
    headers = {"Authorization": f"Token {api_key}"}
    try:
        return await websockets.connect(url, additional_headers=headers)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers)

def _mark_span_error(span) -> None:
    if StatusCode is not None:
        span.set_status(StatusCode.ERROR)

def _words(s: str) -> list[str]:
    return re.sub(r"[^\w\s']", "", s).lower().split()

class CustomVoiceAgent:
    def __init__(
        self, 
        client_websocket, 
        system_prompt: str, 
        max_session_seconds: float,
        inactivity_timeout_seconds: float,
        max_duration_message: str,
        inactivity_message: str,
        greeting_message: str,
        endpointing: int,
        utterance_end: int,
        stable_interim_secs: float,
        stable_interim_secs_no_punct: float,
        model: str,
        stt_model: str,
        tts_model: str,
        tracing: bool,
        session_id: str | None,
        tool_registry: ToolRegistry,
        llm_provider_api_key: str | None = None,
        deepgram_api_key: str | None = None,
        otel_exporters: Sequence[OTLPExporterConfig] | None = None,
        service_name: str = "myvoiceai",
        project_name: str = "myvoiceai-voice-session",
        service_version: str | None = None,
        deployment_environment: str | None = None,
        eval_run_id: str | None = None,
        eval_case_id: str | None = None,
        eval_variant: str | None = None,
    ):
        self.client_ws = client_websocket
        self.tracing = tracing
        self.model = model
        self.llm_provider_api_key = llm_provider_api_key
        self.tool_registry = tool_registry 
        self.session_id = session_id or str(uuid.uuid4())
        self.eval_attributes = {
            key: value
            for key, value in (
                ("eval.run_id", eval_run_id),
                ("eval.case_id", eval_case_id),
                ("eval.variant", eval_variant),
            )
            if value is not None
        }
        self.deepgram_api_key = deepgram_api_key
        self.stt_model = stt_model
        self.tts_model = tts_model
        if not self.deepgram_api_key:
            raise RuntimeError(
                "Pass the Deepgram API key."
            )

        self._tracer_provider = None
        self.tracer = (
            self._initialize_tracer(
                otel_exporters,
                service_name,
                project_name,
                service_version,
                deployment_environment,
            )
            if tracing
            else None
        )

        self.audio_in_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.llm_prompt_queue: asyncio.Queue = asyncio.Queue(maxsize=10)
        self.tts_text_queue: asyncio.Queue = asyncio.Queue(maxsize=200)
        self.audio_out_queue: asyncio.Queue = asyncio.Queue(maxsize=200)

        self.endpointing = endpointing
        self.utterance_end  = utterance_end
        self.stable_interim_secs = stable_interim_secs
        self.stable_interim_secs_no_punct = stable_interim_secs_no_punct

        self.transcript_accumulator = ""
        self.sentence_buffer = ""
        self.is_ai_speaking = False
        self.interruption_event = asyncio.Event()
        self.conversation_history = []
        self._current_turn_span = None
        self.turn_counter = 0
        self.system_prompt = system_prompt + BASE_SYSTEM_PROMPT
        self.greeting_message = greeting_message

        self._tasks: list[asyncio.Task] = []
        self._dg_stt_ws = None
        self._dg_tts_ws = None

        self._last_audio_sent_ts = None
        self._speech_final_ts = None
        self._llm_request_ts = None
        self._tts_first_send_ts = None

        self.max_session_seconds = max_session_seconds
        self.inactivity_timeout_seconds = inactivity_timeout_seconds
        self.max_duration_message = max_duration_message
        self.inactivity_message = inactivity_message
        self._last_activity_ts = time.monotonic()

        self._closing = False
        self._session_end_event = asyncio.Event()
        self._tts_flush_event = asyncio.Event()
        self._playback_complete_event = asyncio.Event()
        self._tts_span_start_ns = None
        self._last_audio_sent_wall_ts = None
        self._any_output_sent = False

        self.is_welcome_message_running = False
        self._active_tool_tasks: list[asyncio.Task] | None = None

        self._interim_text = ""
        self._interim_since_ts = 0.0
        self._stable_words = []
        self._stable_dispatch_ts = 0.0

        self._spoken_text_parts: list[str] = []

    def _initialize_tracer(
        self,
        exporters_config: Sequence[OTLPExporterConfig] | None,
        service_name: str,
        project_name: str,
        service_version: str | None,
        deployment_environment: str | None,
    ):
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
            self._tracer_provider = provider
            return provider.get_tracer("myvoiceai_tracer")
        except Exception:
            logger.warning(
                "Tracing requested but OpenTelemetry SDK/exporter initialization failed; "
                "install myvoiceai[observability] and check the endpoint configuration.",
                exc_info=True,
            )
            return None

    async def run(self):
        """Orchestrates system loops and guarantees cleanup on exit,
        whether that exit is a client disconnect or an unhandled error
        in any one of the loops."""
        self._tasks = [
            asyncio.create_task(self.read_client_mic_loop(), name="mic_in"),
            asyncio.create_task(self.deepgram_stt_loop(), name="stt"),
            asyncio.create_task(self.llm_loop(), name="llm"),
            asyncio.create_task(self.deepgram_tts_loop(), name="tts"),
            asyncio.create_task(self.write_client_speaker_loop(), name="speaker_out"),
            asyncio.create_task(self.session_timer_loop(), name="session_timer"),
        ]
        try:
            await self._send_greeting()
            done, _ = await asyncio.wait(
                self._tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )
            
            for task in done:
                if task.exception():
                    logger.exception(
                        "Task %s failed",
                        task.get_name(),
                        exc_info=task.exception(),
                    )
            
            if self._closing:
                await self._session_end_event.wait()
        except Exception:
            logger.exception("Voice session failed (session_id=%s)", self.session_id)        
        finally:
            await self._shutdown()

    async def _shutdown(self):
        self._cancel_tool_tasks()

        for task in self._tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)

        for ws in (self._dg_stt_ws, self._dg_tts_ws):
            if ws is not None:
                try:
                    await ws.close()
                except Exception:
                    pass
        try:
            await self.client_ws.close()
        except Exception:
            pass        
        if self._tracer_provider is not None:
            try:
                await asyncio.to_thread(self._tracer_provider.shutdown)
            except Exception:
                logger.exception("Failed to shut down session tracer provider.")
        logger.info("Session cleaned up.")

    # adds audio to the audio_in_queue
    async def read_client_mic_loop(self):
        """Task 1: intercepts raw binary audio chunks from the client,
        and dispatches JSON control/ack messages on the same channel."""
        try:
            while True:
                message = await self.client_ws.receive()
                if message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect()

                if "bytes" in message and message["bytes"] is not None:
                    await self.audio_in_queue.put(message["bytes"])
                elif "text" in message and message["text"] is not None:
                    try:
                        data = json.loads(message["text"])
                    except (json.JSONDecodeError, ValueError):
                        continue
                    # IMPORTANT: client has to send the playback_complete event
                    if data.get("control") == "playback_complete":
                        self._last_activity_ts = time.monotonic()
                        self.is_ai_speaking = False
                        self._playback_complete_event.set()
                        self.is_welcome_message_running = False
        except (WebSocketDisconnect, RuntimeError) as e:
            logger.info("Client connection ended (mic loop): %r", e)      

    # sends audio to deepgram and receives text via websocket
    async def deepgram_stt_loop(self):
        """Task 2: full-duplex pipe driving live Deepgram STT."""
        self._dg_stt_ws = await _connect_deepgram(DEEPGRAM_STT_URL.format(stt_model=self.stt_model, endpointing = self.endpointing, utterance_end = self.utterance_end), self.deepgram_api_key)

        dg_ws = self._dg_stt_ws
        logger.info("Connected to Deepgram STT.")

        try:
            await asyncio.gather(
                self._stt_forward_audio(dg_ws),
                self._stt_handle_responses(dg_ws),
                self._stt_stable_interim_watcher(),
            )
        except ConnectionClosed:
            if self._current_turn_span:
                _mark_span_error(self._current_turn_span)
                self._current_turn_span.set_attribute("error.source", "deepgram_stt")
            logger.warning("Deepgram STT connection closed.")

    async def _stt_stable_interim_watcher(self):
        while True:
            await asyncio.sleep(0.05)
            t = self._interim_text
            if not t:
                continue
            punct = t.rstrip().endswith((".", "?", "!"))
            if not punct and len(_words(t)) < 3:
                continue  # fragment like "and"; wait for speech_final / UtteranceEnd
            needed = self.stable_interim_secs if punct else self.stable_interim_secs_no_punct
            if time.monotonic() - self._interim_since_ts >= needed:
                logger.info("Interim stable for >%.1fs, dispatching: %r", needed, t)
                self._stable_words = _words(t)
                self._stable_dispatch_ts = time.monotonic()
                self.transcript_accumulator += f" {t}"
                self._interim_text = ""
                await self._dispatch_final_transcript("stable_interim")

    async def _stt_forward_audio(self, dg_ws):
        while True:
            chunk = await self.audio_in_queue.get()
            await dg_ws.send(chunk)
            self._last_audio_sent_ts = time.monotonic()
            self._last_audio_sent_wall_ts = time.time_ns()
            self.audio_in_queue.task_done()

    async def _dispatch_final_transcript(self, source: str):
        if not self.transcript_accumulator.strip():
            return
        self._last_activity_ts = time.monotonic()
        final_text = self.transcript_accumulator.strip()
        if self._last_audio_sent_ts:
            logger.info(
                "STT time (last audio to %s): %.3fs",
                source, time.monotonic() - self._last_audio_sent_ts
            )
        self._speech_final_ts = time.monotonic()
        logger.info("User: %s", final_text)
        await self._send_client({"transcript": {"role": "user", "text": final_text, "turn_id": self.turn_counter + 1}})
        self.transcript_accumulator = ""
        turn_start_ns = self._last_audio_sent_wall_ts if self._last_audio_sent_wall_ts else time.time_ns()
        self._current_turn_span = self.tracer.start_span(
            "voice_turn",
            attributes={
                "turn.user_text": final_text,
                "turn.id": self.turn_counter + 1,
                "session.id": self.session_id,
                "openinference.span.kind": "CHAIN",
                "input.value": final_text,
                **self.eval_attributes,
            },
        ) if self.tracer else None

        with self._in_turn():
            self._record_span("stt_finalize", turn_start_ns, {
                "openinference.span.kind": "CHAIN",
                "output.value": final_text,
                "stt.source": source,
            })

        result = await check_input(final_text)
        if self._current_turn_span:
            self._current_turn_span.set_attribute("guardrail.input_allowed", result.allowed)
        if not result.allowed:
            self._end_turn_span({"guardrail.input_reason": result.reason or ""})
            await self._speak_guardrail_block()
            return

        await self.llm_prompt_queue.put(final_text)

    def _strip_dispatched(self, text: str) -> str:
        """Drop the part of `text` that was already sent via the stable interim path."""
        if not self._stable_words or time.monotonic() - self._stable_dispatch_ts > 5:
            return text
        n = len(self._stable_words)
        if _words(text)[:n] == self._stable_words:
            return " ".join(text.split()[n:])
        return text    

    async def _stt_handle_responses(self, dg_ws):
        while True:
            try:
                msg = await dg_ws.recv()
                data = json.loads(msg)
            except json.JSONDecodeError:
                logger.warning("Non-JSON message from Deepgram, skipping.")
                continue

            msg_type = data.get("type")
            if msg_type == "UtteranceEnd":
                await self._handle_utterance_end()
            elif msg_type == "Results":
                await self._handle_stt_result(data)

    async def _handle_utterance_end(self):
        logger.info("UtteranceEnd received.")
        if self._interim_text and time.monotonic() - self._interim_since_ts < self.stable_interim_secs_no_punct:
            # Deepgram still has an unfinalized interim; wait for its final
            return
        await self._dispatch_final_transcript("UtteranceEnd")
        self._stable_words = []
        self._interim_text = ""

    async def _handle_stt_result(self, data: dict):
        alt = data.get("channel", {}).get("alternatives", [{}])[0]
        transcript = alt.get("transcript", "")

        if transcript:
            await self._send_client({
                "transcript_chunk": {
                    "role": "user",
                    "turn_id": self.turn_counter + 1,
                    "text": f"{self.transcript_accumulator} {self._interim_text}".strip(),
                    "replace": True,
                }
            })
            self._last_activity_ts = time.monotonic()
            tail = self._strip_dispatched(transcript)
            new_words = [w for w in _words(tail) if w not in FILLERS]

            logger.info("Interim transcript fragment: %r (is_final=%s) (speech_final=%s)", transcript, data.get("is_final"), data.get("speech_final"))

            if self.is_ai_speaking and not self.interruption_event.is_set() and not self._closing and not self.is_welcome_message_running and new_words:
                logger.info("Barge-in detected via transcript: %r", transcript)
                if self._current_turn_span:
                    self._current_turn_span.set_attribute("turn.interrupted_by_barge_in", True)
                self.interruption_event.set()
                self.transcript_accumulator = ""
                await self._purge_pipeline()

            if data.get("is_final"):
                if tail.strip():
                    self.transcript_accumulator += f" {tail}"
                self._interim_text = ""
            elif tail.strip() and tail != self._interim_text:
                self._interim_text = tail
                self._interim_since_ts = time.monotonic()

        if data.get("speech_final"):
            await self._dispatch_final_transcript("speech_final")

    async def llm_loop(self):
        """Task 3: streams prompts to the LLM via LiteLLM, splits the response into
        sentence-sized chunks, and forwards each chunk to the TTS queue
        as soon as it is ready to speak."""
        while True:
    
            prompt = await self.llm_prompt_queue.get()

            if self._closing:
                self.llm_prompt_queue.task_done()
                continue
    
            if self._speech_final_ts:
                logger.info("Time from speech end to LLM dispatch: %.3fs", time.monotonic() - self._speech_final_ts)
            self._llm_request_ts = time.monotonic()
            self._tts_first_send_ts = None
            logger.info("Sending prompt to LLM: %r", prompt)
            self.sentence_buffer = ""
            self._spoken_text_parts = []
            self._any_output_sent = False
    
            self.conversation_history.append(
                {"role": "user", "content": prompt}
            )
    
            messages = self._build_messages()
    
            full_reply = ""

            with self._in_turn(), self._span("llm_stream", {
                "llm.model": self.model,
                "llm.model_name": self.model,
                "openinference.span.kind": "LLM",
                "input.value": prompt,
            }) as llm_span:
                full_reply = await self.llm_call(llm_span=llm_span, messages=messages, prompt=prompt)

            if self._current_turn_span and full_reply.strip():
                self._current_turn_span.set_attribute("output.value", full_reply[:500])
                        
    
          
            if not self._closing:
                if self.sentence_buffer.strip() and not self.interruption_event.is_set():
                    await self._emit_checked_chunk(strip_markdown_for_speech(self.sentence_buffer), flush=False)

                if not self._any_output_sent and not self.interruption_event.is_set():
                    await self._speak_guardrail_block()
                else:
                    await self.tts_text_queue.put({"flush": True})
            self.sentence_buffer = ""
    
            if full_reply.strip() and not self.interruption_event.is_set():
                spoken = " ".join(self._spoken_text_parts).strip() or full_reply
                self.conversation_history.append(
                    {"role": "assistant", "content": full_reply}
                )
                await self._send_client({"transcript": {"role": "assistant", "text": spoken, "turn_id": self.turn_counter + 1}})
                self.turn_counter += 1
                await self._send_client({
                    "turn": {
                        "turn_id": self.turn_counter,
                        "timestamp": time.time(),
                        "user": prompt,
                        "assistant": spoken,
                        "interrupted": False,
                    }
                })
    
            self.llm_prompt_queue.task_done()

    async def _buffer_and_dispatch(self, token: str):
        self.sentence_buffer += token
    
        flush_idx = -1
        for i, ch in enumerate(self.sentence_buffer):
            if ch in SENTENCE_BOUNDARY_CHARS:
                flush_idx = i
            elif ch in CLAUSE_BOUNDARY_CHARS and i > 40:
                flush_idx = i
    
        if flush_idx != -1:
            text_to_send = strip_markdown_for_speech(self.sentence_buffer[:flush_idx + 1]).strip()
            self.sentence_buffer = self.sentence_buffer[flush_idx + 1:]
        elif len(self.sentence_buffer) > MAX_BUFFER_CHARS_BEFORE_FORCED_FLUSH:
            text_to_send = strip_markdown_for_speech(self.sentence_buffer).strip()
            self.sentence_buffer = ""
        else:
            return

        if text_to_send:
            await self._emit_checked_chunk(text_to_send, flush=True)

    async def _emit_checked_chunk(self, text: str, flush: bool):
        """Runs the output guardrail on a chunk of the reply and, if allowed,
        queues it for TTS and shows it in the client transcript."""
        result = await check_output(text)
        if self._current_turn_span:
            self._current_turn_span.set_attribute("guardrail.output_allowed", result.allowed)
        if not result.allowed:
            logger.info("Output guardrail blocked chunk: %s", result.reason)
            return
        logger.info("Flushing to tts_text_queue at t=%.3f: %r", time.monotonic(), text)
        await self.tts_text_queue.put(text)
        if flush:
            await self.tts_text_queue.put({"flush": True})
        self._any_output_sent = True
        await self._emit_agent_chunk(text)


    async def llm_call(self, messages, prompt, llm_span=None) -> str:
        full_reply = ""
        if llm_span:
            llm_span.set_attribute("openinference.span.kind", "LLM")
            llm_span.set_attribute("input.value", prompt)
        try:
            current_messages = messages
            hop = 0
            while hop < MAX_TOOL_HOPS:
                reply_part, tool_calls = await self._stream_completion(current_messages)
                full_reply += reply_part

                if self.interruption_event.is_set():
                    self.conversation_history.append({"role": "assistant", "content": "No response."})
                    self.turn_counter += 1
                    await self._send_client({
                        "turn": {
                            "turn_id": self.turn_counter, "timestamp": time.time(),
                            "user": prompt, "assistant": full_reply if full_reply.strip() else None,
                            "interrupted": True,
                        }
                    })
                    break

                if not tool_calls:
                    break

                ok = await self._execute_tool_hop(tool_calls)
                if not ok:
                    break
                current_messages = self._build_messages()
                hop += 1
                
            if llm_span:
                llm_span.set_attribute("llm.response_text", full_reply[:500])
                llm_span.set_attribute("output.value", full_reply[:500])
            logger.info("LLM total time to final token: %.3fs", time.monotonic() - (self._llm_request_ts or 0))

        except exceptions.APIError:
            if llm_span:
                llm_span.record_exception(traceback_exc := __import__("sys").exc_info()[1])
            if StatusCode and llm_span:
                llm_span.set_status(StatusCode.ERROR)
            logger.exception("LLM request failed.")
        except Exception:
            if llm_span:
                llm_span.record_exception(__import__("sys").exc_info()[1])
            if StatusCode and llm_span:    
                llm_span.set_status(StatusCode.ERROR)
            logger.exception("Unexpected error during LLM streaming.")
        return full_reply                             
       

    async def deepgram_tts_loop(self):
        """Task 4: submits buffered text to Deepgram's Aura streaming TTS
        and streams the resulting PCM audio back out."""
        self._dg_tts_ws = await _connect_deepgram(DEEPGRAM_TTS_URL.format(tts_model=self.tts_model), self.deepgram_api_key)
        tts_ws = self._dg_tts_ws

        try:
            await asyncio.gather(self._tts_feed_text(tts_ws), self._tts_harvest_audio(tts_ws))
        except ConnectionClosed:
            logger.warning("Deepgram TTS connection closed.")

    async def _tts_feed_text(self, tts_ws):
        while True:
            item = await self.tts_text_queue.get()
            logger.info("feed_text_to_tts dequeued at t=%.3f: %r", time.monotonic(), item)
            if isinstance(item, dict) and item.get("flush"):
                logger.info("Sending Flush to Deepgram TTS.")
                await tts_ws.send(json.dumps({"type": "Flush"}))
            elif not self.interruption_event.is_set():
                if self._tts_first_send_ts is None:
                    self._tts_first_send_ts = time.monotonic()
                    self._tts_span_start_ns = time.time_ns()
                with self._in_turn(), self._span("tts_stream", {
                    "tts.text_len": len(item),
                    "openinference.span.kind": "CHAIN",
                    "input.value": item,
                }, require_turn=True):
                    await tts_ws.send(json.dumps({"type": "Speak", "text": item}))
                logger.info("Sending text to Deepgram TTS at t=%.3f: %r", time.monotonic(), item)
            self.tts_text_queue.task_done()

    async def _tts_harvest_audio(self, tts_ws):
        while True:
            msg = await tts_ws.recv()
            if isinstance(msg, bytes) and not self.interruption_event.is_set():
                if self._tts_first_send_ts is not None:
                    logger.info("TTS time to first audio: %.3fs", time.monotonic() - self._tts_first_send_ts)
                    if self._current_turn_span and self._tts_span_start_ns:
                        with self._in_turn():
                            self._record_span("tts_generation", self._tts_span_start_ns, {
                                "openinference.span.kind": "CHAIN",
                            })
                    self._tts_first_send_ts = None
                    self._tts_span_start_ns = None
                await self.audio_out_queue.put(msg)
            else:
                try:
                    data = json.loads(msg)
                    if data.get("type") == "Flushed":
                        await self.audio_out_queue.put(_AUDIO_UTTERANCE_END)
                        self._tts_flush_event.set()
                        self._end_turn_span()
                except json.JSONDecodeError:
                    pass

    async def write_client_speaker_loop(self):
        while True:
            audio_payload = await self.audio_out_queue.get()
            try:
                if audio_payload is _AUDIO_UTTERANCE_END:
                    await self.client_ws.send_json({"control": "utterance_end"})
                    continue

                if not self.interruption_event.is_set():
                    self.is_ai_speaking = True
                    await self.client_ws.send_bytes(audio_payload)
            except WebSocketDisconnect:
                logger.info("Client disconnected (speaker loop).")
                return
            except RuntimeError:
                logger.info("Client websocket not ready or closed (speaker loop).")
                return
            finally:
                self.audio_out_queue.task_done()

    async def session_timer_loop(self):
        """Task 6: enforces a max session duration and an inactivity timeout,
        closing the session with a spoken message when either is hit."""
        session_start = time.monotonic()
        while True:
            await asyncio.sleep(1)
    
            now = time.monotonic()
    
            if now - session_start > self.max_session_seconds:
                logger.info("Max session duration (%.0fs) reached, closing.", self.max_session_seconds)
                await self._say_goodbye_and_close(self.max_duration_message)
                return
    
            if (now - self._last_activity_ts > self.inactivity_timeout_seconds) and not self.is_ai_speaking:
                logger.info("Inactivity timeout (%.0fs) reached, closing.", self.inactivity_timeout_seconds)
                await self._say_goodbye_and_close(self.inactivity_message)
                return

    async def _say_goodbye_and_close(self, text: str):
        """Halts any in-flight LLM/TTS output, then speaks a final message
        through the pipeline and waits for it to finish playing."""
        self._closing = True
        self._tts_flush_event.clear()
        self._playback_complete_event.clear()
        logger.info("Executing goodbye sequence: %r", text)

        self.interruption_event.set()
        await self._purge_pipeline()

        # 3. Ensure state triggers let the text pass through cleanly
        self.interruption_event.clear() 
        self.is_ai_speaking = True  # Hold this True so the loop knows audio is expected
    
        # 4. Enqueue the final text
        await self._speak(text)
        
        # 5. Wait for the audio to generate, land in the queue, and finish playing
        # Wait a brief moment for the generator to catch up before checking if queues are empty
        await asyncio.sleep(0.3) 
        
        try:
            await asyncio.wait_for(
                self._tts_flush_event.wait(),
                timeout=DEFAULT_GOODBYE_WAIT_SECONDS,
            )
            logger.info("Deepgram finished generating goodbye audio.")
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for Deepgram goodbye audio.")

        try:
            await asyncio.wait_for(
                self.audio_out_queue.join(),
                timeout=DEFAULT_GOODBYE_WAIT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.warning("Timed out sending goodbye audio to client.")

        try:
            await asyncio.wait_for(
                self._playback_complete_event.wait(),
                timeout=DEFAULT_GOODBYE_WAIT_SECONDS,
            )
            logger.info("Client finished playing goodbye audio.")
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for goodbye playback acknowledgement.")

        self._session_end_event.set()    
    

    async def _purge_pipeline(self):
        """Drains in-flight queues instantly during a user barge-in and
        tells the browser to drop whatever it has already buffered."""
        # Purge AI output only; interrupted user audio (audio_in_queue)
        # must stay alive so the interrupted turn becomes the new prompt.
        self._end_turn_span({"turn.interrupted": True})
        self._cancel_tool_tasks()

        for q in (self.llm_prompt_queue, self.tts_text_queue, self.audio_out_queue):
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except asyncio.QueueEmpty:
                    break
                
        self.is_ai_speaking = False
        self.interruption_event.clear()
        await self._send_client({"control": "clear_speaker_buffer"})
        if self._dg_tts_ws is not None:
            try:
                await self._dg_tts_ws.send(json.dumps({"type": "Clear"}))
            except Exception:
                pass

    async def _emit_agent_chunk(self, text: str):
        self._spoken_text_parts.append(text)
        await self._send_client({
            "transcript_chunk": {
                "role": "assistant",
                "turn_id": self.turn_counter + 1,
                "text": text,
            }
        })

    async def _send_client(self, payload: dict):
        """Best-effort JSON message to the browser; a closed socket is ignored."""
        try:
            await self.client_ws.send_json(payload)
        except Exception:
            pass

    async def _speak(self, text: str):
        """Queues a complete message for TTS, followed by a flush."""
        await self.tts_text_queue.put(strip_markdown_for_speech(text))
        await self.tts_text_queue.put({"flush": True})

    def _cancel_tool_tasks(self):
        if self._active_tool_tasks is not None:
            for t in self._active_tool_tasks:
                if not t.done():
                    t.cancel()
            self._active_tool_tasks = None

    def _end_turn_span(self, attributes: dict | None = None):
        if self._current_turn_span:
            for key, value in (attributes or {}).items():
                self._current_turn_span.set_attribute(key, value)
            self._current_turn_span.end()
            self._current_turn_span = None

    def _build_messages(self) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt}] + self.conversation_history

    def _in_turn(self):
        """Makes the current turn span the parent of spans started inside it."""
        if self._current_turn_span is None or trace is None:
            return contextlib.nullcontext()
        return trace.use_span(self._current_turn_span, end_on_exit=False)

    def _span(self, name: str, attributes: dict, require_turn: bool = False):
        """Starts a span as the current span; yields None when tracing is off
        (or, with require_turn, when no turn span is open)."""
        if self.tracer is None or (require_turn and self._current_turn_span is None):
            return contextlib.nullcontext()
        return self.tracer.start_as_current_span(name, attributes=attributes)

    def _record_span(self, name: str, start_time: int, attributes: dict):
        """Records a span that started at start_time and ends now."""
        if self.tracer:
            self.tracer.start_span(name, start_time=start_time, attributes=attributes).end()

    async def _speak_guardrail_block(self):
        self.is_ai_speaking = True
        await self._speak(GUARDRAIL_BLOCK_MESSAGE)

    async def _send_greeting(self):
        self.is_welcome_message_running = True
        self.is_ai_speaking = True
        await self._speak(self.greeting_message)

    async def _run_tool_call(self, name: str, args: dict) -> dict:
        impl = self.tool_registry.lookup(name)
        if impl is None:
            return {"error": f"unknown tool: {name}"}
        serialized_args = json.dumps(args, default=str) if self.tracer else ""
        with self._span("tool_call", {
            "tool.name": name,
            "tool.args": serialized_args,
            "openinference.span.kind": "TOOL",
            "input.value": serialized_args,
        }) as span:
            try:
                result = await impl(**args)
                if span:
                    serialized_result = json.dumps(result, default=str)[:500]
                    span.set_attribute("tool.result", serialized_result)
                    span.set_attribute("output.value", serialized_result)
                return result
            except asyncio.CancelledError:
                if span:
                    span.set_attribute("tool.cancelled", True)
                raise
            except Exception as e:
                if span:
                    span.record_exception(e)
                    _mark_span_error(span)
                return {"error": str(e)}


    async def _stream_completion(self, messages: list) -> tuple[str, dict[int, dict]]:
        """Runs one streaming completion pass. Returns (text_reply, tool_call_accumulator)."""
        reply = ""
        tool_call_accumulator: dict[int, dict] = {}
        first_token_logged = False

        response = cast(CustomStreamWrapper, await acompletion(
            model=self.model,
            messages=messages,
            stream=True,
            timeout=30.0,
            api_key=self.llm_provider_api_key,
            tools=self.tool_registry.schemas() or None,
        ))

        async for chunk in response:
            if self.interruption_event.is_set():
                break

            try:
                text_token = chunk.choices[0].delta.content
            except (AttributeError, IndexError):
                text_token = None

            if text_token:
                reply += text_token
                if not first_token_logged:
                    logger.info("LLM time to first token: %.3fs", time.monotonic() - (self._llm_request_ts or 0.0))
                    first_token_logged = True
                await self._buffer_and_dispatch(text_token)

            try:
                delta_tool_calls = chunk.choices[0].delta.tool_calls
            except (AttributeError, IndexError):
                delta_tool_calls = None

            if delta_tool_calls:
                for tc in delta_tool_calls:
                    idx = tc.index
                    entry = tool_call_accumulator.setdefault(idx, {"id": None, "name": None, "arguments": ""})
                    if tc.id:
                        entry["id"] = tc.id
                    function = getattr(tc, "function", None)
                    if function and getattr(function, "name", None):
                        entry["name"] = function.name
                    if function and getattr(function, "arguments", None):
                        entry["arguments"] += function.arguments

            if chunk.choices[0].finish_reason == "tool_calls":
                break

        return reply, tool_call_accumulator     


    async def _execute_tool_hop(self, tool_call_accumulator: dict[int, dict]) -> bool:
        self.is_ai_speaking = True
        await self._speak(random.choice(TOOL_FILLER_PHRASES))

        entries = list(tool_call_accumulator.values())

        assistant_tool_calls = [
            {
                "id": entry["id"], "type": "function",
                "function": {"name": entry["name"], "arguments": entry["arguments"]},
            }
            for entry in entries
        ]
        self.conversation_history.append({"role": "assistant", "tool_calls": assistant_tool_calls})

        parsed_args = []
        for entry in entries:
            try:
                parsed_args.append(json.loads(entry["arguments"] or "{}"))
            except json.JSONDecodeError:
                parsed_args.append({})

        tool_tasks = [
            asyncio.create_task(self._run_tool_call(entry["name"], args))
            for entry, args in zip(entries, parsed_args)
        ]
        self._active_tool_tasks = tool_tasks

        tools_group = asyncio.gather(*tool_tasks, return_exceptions=True)
        interrupt_wait_task = asyncio.create_task(self.interruption_event.wait())

        done, _ = await asyncio.wait(
            [tools_group, interrupt_wait_task],
            timeout=TOOL_CALL_TIMEOUT_SECONDS,
            return_when=asyncio.FIRST_COMPLETED,
        )

        interrupt_wait_task.cancel()

        if interrupt_wait_task in done or tools_group not in done:
            # interrupted, or timed out
            self._cancel_tool_tasks()
            if interrupt_wait_task not in done:
                await self._speak("That's taking longer than expected, let me get back to you.")
            return False

        self._active_tool_tasks = None

        if self.interruption_event.is_set():
            return False

        for entry, task in zip(entries, tool_tasks):
            if task.cancelled():
                continue
            try:
                tool_result = task.result()
            except Exception as e:
                tool_result = {"error": str(e)}
            self.conversation_history.append({
                "role": "tool", "tool_call_id": entry["id"], "content": json.dumps(tool_result),
            })

        return True      

async def run_voice_session(
    websocket,
    *,
    system_prompt=SYSTEM_PROMPT,
    greeting_message=DEFAULT_GREETING_MESSAGE,
    inactivity_message: str = INACTIVITY_MESSAGE,
    max_duration_message: str = MAX_DURATION_MESSAGE,
    max_session_seconds: float = DEFAULT_MAX_SESSION_SECONDS,
    endpointing=DEFAULT_ENDPOINTING,
    utterance_end=DEFAULT_UTTERANCE_END,
    stable_interim_secs=STABLE_INTERIM_SECS,
    stable_interim_secs_no_punct=STABLE_INTERIM_NO_PUNCT_SECS,
    inactivity_timeout_seconds=DEFAULT_INACTIVITY_TIMEOUT_SECONDS,
    model=LLM_MODEL,
    llm_provider_api_key=None,
    tracing=False,
    session_id: str | None = None,
    deepgram_api_key=None,
    stt_model=DEEPGRAM_STT_MODEL,
    tts_model=DEEPGRAM_TTS_MODEL,
    tool_registry=get_default_registry(),
    otel_exporters: Sequence[OTLPExporterConfig] | None = None,
    service_name: str = "myvoiceai",
    project_name: str = "myvoiceai-voice-session",
    service_version: str | None = None,
    deployment_environment: str | None = None,
    eval_run_id: str | None = None,
    eval_case_id: str | None = None,
    eval_variant: str | None = None,
    **kwargs
):  
    if kwargs:
        logger.warning("run_voice_session: ignoring unsupported options: %s", ", ".join(sorted(kwargs)))  
    try:
        agent = CustomVoiceAgent(
            client_websocket=websocket,
            system_prompt=system_prompt,
            greeting_message=greeting_message,
            endpointing=endpointing,
            utterance_end=utterance_end,
            stable_interim_secs=stable_interim_secs,
            stable_interim_secs_no_punct=stable_interim_secs_no_punct,
            inactivity_timeout_seconds=inactivity_timeout_seconds,
            session_id=session_id,
            deepgram_api_key=deepgram_api_key,
            model=model,
            llm_provider_api_key=llm_provider_api_key,
            tracing=tracing,
            tool_registry=tool_registry,
            stt_model=stt_model,
            tts_model=tts_model,
            max_session_seconds=max_session_seconds,
            max_duration_message=max_duration_message,
            inactivity_message=inactivity_message,
            otel_exporters=otel_exporters,
            service_name=service_name,
            project_name=project_name,
            service_version=service_version,
            deployment_environment=deployment_environment,
            eval_run_id=eval_run_id,
            eval_case_id=eval_case_id,
            eval_variant=eval_variant,
        )
        await agent.run()
    except asyncio.CancelledError:
        raise    
    except Exception:
        logger.exception("Voice session failed for session_id=%s", session_id)
        try:
            await websocket.close(code=1011, reason="Session error")
        except Exception:
            pass
