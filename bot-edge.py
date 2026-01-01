"""Pipecat Local "Edge-Latency" Voice Agent (bot-edge.py).

This bot implements the SOTA "Edge-Latency" proposal, strictly local, optimized for:
1. True real-time transcription (Interim + Final updates).
2. Hybrid EOT (VAD + Semantic Turn Detection).
3. Real-time punctuation/correction policy.
4. Faster TTS start (Micro-chunk aggregation).

See spec.md for architectural details.
"""
import asyncio
import os
import sys
import time
import wave
import io
import re
import string
import numpy as np
from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    AudioRawFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    LLMRunFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
)
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.stt_service import STTService
from pipecat.services.whisper.stt import Model, resolve_language
from pipecat.services.xtts.tts import XTTSService
from pipecat.services.resembleai.tts import ResembleAITTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.runner.types import RunnerArguments, SmallWebRTCRunnerArguments
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.text.base_text_aggregator import Aggregation, AggregationType
from pipecat.utils.string import match_endofsentence
from pipecat.utils.time import time_now_iso8601

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

# --- Configuration (from spec) ---
STT_MODE = os.getenv("STT_MODE", "streaming")  # streaming | segmented
STT_INTERIM_INTERVAL_MS = int(os.getenv("STT_INTERIM_INTERVAL_MS", "200"))
VAD_STOP_SECS = float(os.getenv("VAD_STOP_SECS", "0.8"))
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "instructions.md")

# --- 1. Custom Aggregator (Low Latency) ---
class SpaceAwareTextAggregator(SimpleTextAggregator):
    """Aggregator that attempts to yield sentences eagerly on whitespace/newline."""
    async def _check_sentence_with_lookahead(self, char: str):
        if self._needs_lookahead and (char.isspace() or char == "\n"):
             eos_marker = match_endofsentence(self._text)
             if eos_marker:
                 result = self._text[:eos_marker]
                 self._text = self._text[eos_marker:]
                 self._needs_lookahead = False
                 return Aggregation(text=result.strip(), type=AggregationType.SENTENCE)
        return await super()._check_sentence_with_lookahead(char)

# --- 2. Local Streaming STT Service (Option A) ---
class LocalStreamingWhisperSTTService(STTService):
    """
    Local STT that simulates streaming by periodically re-transcribing the growing buffer.
    Emits InterimTranscriptionFrame frequently, and TranscriptionFrame on VAD flush.
    """
    def __init__(
        self,
        *,
        model: str | Model = Model.LARGE_V3_TURBO,
        device: str = "auto",
        compute_type: str = "default",
        no_speech_prob: float = 0.4,
        language="en",
        interim_interval_ms: int = 300,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._device = device
        self._compute_type = compute_type
        self.set_model_name(model if isinstance(model, str) else model.value)
        self._no_speech_prob = no_speech_prob
        self._language = language
        self._interim_interval_ms = interim_interval_ms
        self._model = None
        
        # Buffer state
        self._audio_buffer = bytearray()
        self._last_interim_time = 0
        self._is_streaming = False
        self._current_user_id = ""
        
        # Lazy load in start()
        # self._load() 

    async def start(self, frame: StartFrame):
        asyncio.create_task(self._load_model())
        await super().start(frame)

    async def run_stt(self, audio: bytes):
        # We override process_audio_frame, so this is just to satisfy the abstract base class.
        yield None

    async def _load_model(self):
        if not self._model:
            logger.info(f"Loading Whisper Model: {self.model_name} on {self._device}...")
            self._model = await asyncio.to_thread(self._load_sync)
            logger.info("Whisper model loaded.")

    def _load_sync(self):
        try:
            from faster_whisper import WhisperModel
            return WhisperModel(
                self.model_name, device=self._device, compute_type=self._compute_type
            )
        except Exception as e:
            logger.error(f"Failed to load Whisper: {e}")
            return None

    async def process_audio_frame(self, frame: AudioRawFrame, direction: FrameDirection):
        # Capture user ID if present
        if hasattr(frame, "user_id"):
            self._current_user_id = frame.user_id
        
        # Continuously buffer audio
        if self._is_streaming or len(self._audio_buffer) > 0:
             self._audio_buffer += frame.audio
             await self._check_interim()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, VADUserStartedSpeakingFrame):
            logger.debug("VAD: User started speaking")
            self._is_streaming = True
            
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            logger.debug("VAD: User stopped speaking -> Finalizing")
            self._is_streaming = False
            await self._finalize_transcription()

    async def _check_interim(self):
        now = time.time() * 1000
        if now - self._last_interim_time > self._interim_interval_ms:
            self._last_interim_time = now
            if len(self._audio_buffer) > 0:
                await self._transcribe_buffer(is_final=False)

    async def _finalize_transcription(self):
        if len(self._audio_buffer) > 0:
            await self._transcribe_buffer(is_final=True)
            self._audio_buffer.clear()

    async def _transcribe_buffer(self, is_final: bool):
        if not self._model: 
            return

        # Prepare audio
        audio_data = bytes(self._audio_buffer)
        
        # Run inference in thread
        text = await asyncio.to_thread(self._run_inference_sync, audio_data)
        
        if text:
            timestamp = time_now_iso8601()
            # If final, emit TranscriptionFrame
            if is_final:
                logger.debug(f"STT Final: {text}")
                frame = TranscriptionFrame(
                    text=text,
                    user_id=self._current_user_id,
                    timestamp=timestamp,
                    language=None
                )
                await self.push_frame(frame)
            else:
                # If interim, emit InterimTranscriptionFrame
                # Only if text length changed or substantial
                frame = InterimTranscriptionFrame(
                    text=text,
                    user_id=self._current_user_id,
                    timestamp=timestamp,
                    language=None
                )
                await self.push_frame(frame)

    def _run_inference_sync(self, audio_data: bytes) -> str:
        # Convert to float32
        audio_float = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        segments, _ = self._model.transcribe(
            audio_float, 
            language=self._language,
            beam_size=1, # Fast greedy for interim
            # If is_final=True we could use larger beam, but for consistency stick to 1 or config
        )
        # Collect text
        text = " ".join([s.text for s in segments]).strip()
        return text

# --- 3. Transcription Policy Processor ---
class TranscriptionPolicyProcessor(FrameProcessor):
    """
    Applies real-time punctuation/correction policy:
    - Interim: Lowercase, minimal punctuation, no 'clean up' (fast).
    - Final: Proper punctuation and capitalization.
    """
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterimTranscriptionFrame):
            # Policy: Light normalization for UI stability
            # Avoid heavy re-formatting that causes "jumping"
            # Here we just pass it through, or maybe strip trailing punctuation if it flickers
            frame.text = self._apply_interim_policy(frame.text)
            await self.push_frame(frame, direction)
            
        elif isinstance(frame, TranscriptionFrame):
            # Policy: Stronger normalization could happen here if not done by model.
            # Whisper usually does good punctuation on Final.
            logger.debug(f"Policy Final: {frame.text}")
            await self.push_frame(frame, direction)
            
        else:
            await super().process_frame(frame, direction)
            await self.push_frame(frame, direction)

    def _apply_interim_policy(self, text: str) -> str:
        # Simple policy: just trim
        return text.strip()

class InterimBlockingFilter(FrameProcessor):
    """Blocks InterimTranscriptionFrames from passing further down the pipeline."""
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if isinstance(frame, InterimTranscriptionFrame):
            return
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

# --- Main Bot Flow ---
async def bot(runner_args: RunnerArguments):
    webrtc_connection = None
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        webrtc_connection = runner_args.webrtc_connection
    else:
        logger.error(f"Expected SmallWebRTCRunnerArguments, got {type(runner_args)}")
        return

    # 1. Transport with VAD
    # Using Silero VAD with tuned STOP_SECS (0.8s default in spec)
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_in_enabled=True,
            camera_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=VAD_STOP_SECS)),
        )
    )

    logger.info("Initializing LocalStreamingWhisperSTTService (lazy load)...")
    # 2. STT (Local Streaming)
    stt = LocalStreamingWhisperSTTService(
        model=Model.LARGE_V3_TURBO,
        device="cuda",
        interim_interval_ms=STT_INTERIM_INTERVAL_MS,
    )
    logger.info("LocalStreamingWhisperSTTService initialized (lazy).")
    
    # 3. Policy Processor
    policy_processor = TranscriptionPolicyProcessor()

    # 4. LLM Service
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "qwen2.5:14b"),
        params=BaseOpenAILLMService.InputParams(
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        ),
    )

    # 5. TTS Service (Fast start)
    import aiohttp
    async with aiohttp.ClientSession() as session:
        # Use SpaceAwareTextAggregator for lower latency (micro-chunks)
        text_aggregator = SpaceAwareTextAggregator()
        
        tts_service_env = os.getenv("TTS_SERVICE", "xtts")
        if tts_service_env == "chatterbox":
            tts = ResembleAITTSService(
                api_key="dummy",
                voice_uuid=os.getenv("TTS_VOICE", "43c3da60-7604-4b80-827d-08b52822452a"),
                url=os.getenv("CHATTERBOX_URL", "ws://localhost:8004/stream"),
                sample_rate=22050,
                aggregate_sentences=True,
                text_aggregator=text_aggregator,
            )
        else:
            tts = XTTSService(
                voice_id=os.getenv("TTS_VOICE", "Claribel Dervla"),
                base_url="http://localhost:8000",
                aiohttp_session=session,
                text_aggregator=text_aggregator,
            )

        # 6. Context Management (System Prompt + History)
        system_instructions = "You are a helpful assistant."
        try:
            with open(SYSTEM_PROMPT_PATH, "r") as f:
                system_instructions = f.read()
        except Exception:
            pass

        messages = [{"role": "system", "content": system_instructions}]
        context = LLMContext(messages)
        
        # SOTA Turn Strategy: Hybrid (VAD stop hard, but Semantic check for "hard stop")
        # Spec 4.4: "Only let VAD produce early soft stop; let semantic stop confirm hard stop"
        # However, calling context_aggregator.user() triggers checking strategies.
        # Use LocalSmartTurnAnalyzerV3.
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_stop_timeout=1.2, # Spec 4.4: Increase fallback timeout
                user_turn_strategies=UserTurnStrategies(
                    stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
                ),
            ),
        )

        # 7. RTVI (Realtime Frontend)
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # 8. Pipeline Construction (Canonical Order from Spec 5.1)
        # 1. transport.input()
        # 2. rtvi (processor)
        # 3. stt_streaming (emit interims + finals)
        # 4. transcription_policy_processor
        # 5. context_aggregator.user() (Consumes User Turns)
        # 6. llm
        # 7. tts
        # 8. transport.output()
        # 9. context_aggregator.assistant()
        
        pipeline_steps = [
            transport.input(),
            rtvi,            # Inspects frames, handles client messages
            stt,             # Produces InterimTranscriptionFrame + TranscriptionFrame
            policy_processor,# Modifies frames
            InterimBlockingFilter(), # Blocks Interim frames from Context
            context_aggregator.user(), # Consumes TranscriptionFrame (final), ignores Interim?
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]

        pipeline = Pipeline(pipeline_steps)

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)], # Watches for all frames to send to UI
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()
            await task.queue_frames([RTVIServerMessageFrame(data={"show_text_container": True})])
            # Initial greeting
            messages.append({"role": "system", "content": "Please introduce yourself."})
            await task.queue_frames([LLMRunFrame()])

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            logger.info(f"Client connected: {client}")

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await task.queue_frames([EndFrame()])

        runner = PipelineRunner()
        await runner.run(task)

if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
