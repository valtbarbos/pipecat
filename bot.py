"""SOTA Pipecat Local Bot Implementation.

This bot demonstrates a State-Of-The-Art (SOTA) conversational AI pipeline running
entirely on local hardware (optimized for RTX 4090).

Features:
- STT: Whisper Large V3 Turbo (Local GPU via faster-whisper)
- LLM: Qwen 2.5 14B (Local via Ollama)
- TTS: Coqui XTTS v2 (Local)
- VAD: SileroVAD (Tuned to 0.3s for real-time transcription feedback)
- Turn Detection: LocalSmartTurnAnalyzerV3 (Semantic end-of-turn detection)
- UI: RTVI-compatible WebRTC interface

Usage:
  python bot.py -t webrtc --host 0.0.0.0 --port 7860
"""
import os
import sys

from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import (
    RTVIConfig,
    RTVIObserver,
    RTVIProcessor,
    RTVIServerMessageFrame,
)
from pipecat.services.openai.base_llm import BaseOpenAILLMService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.services.xtts.tts import XTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.user_stop import TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.runner.types import SmallWebRTCRunnerArguments, RunnerArguments
from pipecat.utils.text.base_text_aggregator import BaseTextAggregator, Aggregation, AggregationType
from pipecat.utils.text.simple_text_aggregator import SimpleTextAggregator
from pipecat.utils.string import match_endofsentence

class SpaceAwareTextAggregator(SimpleTextAggregator):
    """Aggregator that attempts to yield sentences eagerly on whitespace/newline.
    
    Standard SimpleTextAggregator waits for a non-whitespace character to confirm
    a sentence boundary (lookahead). This adds latency equal to the generation time
    of the first token of the NEXT sentence.
    
    This aggregator checks for sentence completeness as soon as a space or newline 
    is encountered, allowing audio to start much sooner.
    """
    async def _check_sentence_with_lookahead(self, char: str):
        if self._needs_lookahead and (char.isspace() or char == "\n"):
             eos_marker = match_endofsentence(self._text)
             if eos_marker:
                 result = self._text[:eos_marker]
                 self._text = self._text[eos_marker:]
                 self._needs_lookahead = False
                 return Aggregation(text=result.strip(), type=AggregationType.SENTENCE)
        
        return await super()._check_sentence_with_lookahead(char)


# Try importing Mem0, handle rejection if missing (though we added to Dockerfile)
try:
    from pipecat.services.mem0.memory import Mem0MemoryService
    MEM0_AVAILABLE = True
except ImportError:
    logger.warning("Mem0 not installed. Memory features will be disabled.")
    MEM0_AVAILABLE = False

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

async def bot(runner_args: RunnerArguments):
    webrtc_connection = None
    
    # 1. Transport Setup (WebRTC)
    if isinstance(runner_args, SmallWebRTCRunnerArguments):
        webrtc_connection = runner_args.webrtc_connection
    else:
        logger.error(f"Expected SmallWebRTCRunnerArguments, got {type(runner_args)}")
        return

    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            camera_in_enabled=True,
            camera_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.3)), # Aggressive VAD for faster transcription feedback
        )
    )

    # 2. Services Setup
    logger.info("Initializing Whisper STT service...")
    # STT: Local Whisper (running on GPU in this container)
    # Using Large V3 Turbo for SOTA punctuation and accuracy on RTX 4090
    # WRAPPING IN THREAD prevents blocking the event loop during model download/load!
    import asyncio
    stt = await asyncio.to_thread(
        lambda: WhisperSTTService(model=Model.LARGE_V3_TURBO, device="cuda", no_speech_prob=0.4)
    )
    logger.info("Whisper STT service initialized.")

    # LLM: Ollama (Local) running Qwen 2.5 14B
    # Note: Ensure you have run `ollama pull qwen2.5:14b`
    logger.info("Initializing LLM service...")
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "qwen2.5:14b"),
        params=BaseOpenAILLMService.InputParams(
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "512")),
        ),
    )
    logger.info("LLM service initialized.")

    # TTS: XTTS (Local)
    import aiohttp
    async with aiohttp.ClientSession() as session:
        logger.info("Initializing TTS service...")
        tts = XTTSService(
            voice_id=os.getenv("TTS_VOICE", "Claribel Dervla"),
            base_url="http://localhost:8000",
            aiohttp_session=session,
            text_aggregator=SpaceAwareTextAggregator(),
        )
        logger.info("TTS service initialized.")

        # 3. Memory Setup (Mem0)
        memory = None
        if MEM0_AVAILABLE:
            # Note: For strict local usage, you would configure a local qdrant/chroma here.
            # This default setup assumes MEM0_API_KEY might be present or uses defaults.
            # To go fully local with Mem0, you would un-comment and configure the 'local_config'.
            
            # local_mem0_config = {
            #     "vector_store": {
            #         "provider": "qdrant",
            #         "config": {
            #             "host": "localhost",
            #             "port": 6333
            #         }
            #     },
            #     "llm": {
            #         "provider": "openai",
            #         "config": {
            #             "model": "gemma3:27b",
            #             "openai_base_url": "http://localhost:11434/v1",
            #             "api_key": "ollama"
            #         }
            #     }
            # }

            if os.getenv("MEM0_API_KEY"):
                memory = Mem0MemoryService(
                    api_key=os.getenv("MEM0_API_KEY"),
                    user_id="local_user_4090",
                )
                logger.info("Mem0 Memory Service initialized.")
            else:
                logger.warning("MEM0_API_KEY not found. Skipping memory layer for this run.")

        # 4. Context & Flow Control
        # SOTA System Prompt
        # 4. Context & Flow Control
        # SOTA System Prompt
        system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "instructions.md")
        system_instructions = ""
        try:
            with open(system_prompt_path, "r") as f:
                system_instructions = f.read()
            logger.info(f"Loaded system instructions from {system_prompt_path}")
        except FileNotFoundError:
            logger.error(f"System prompt file not found at {system_prompt_path}. Using fallback.")
            system_instructions = (
                "You are a helpful assistant."
            )

        messages = [
            {
                "role": "system",
                "content": system_instructions,
            }
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                user_turn_stop_timeout=0.5, # Fallback if strategies don't trigger
                user_turn_strategies=UserTurnStrategies(
                    # SOTA: Analyzing the meaning of the turn (complete vs incomplete)
                    stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
                ),
            ),
        )
        
        # 5. RTVI (Frontend Feedback)
        rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

        # 6. Pipeline Construction
        pipeline_steps = [
            transport.input(),
            rtvi,
            stt,
            context_aggregator.user(),
        ]
        
        if memory:
            pipeline_steps.append(memory)
            
        pipeline_steps.extend([
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ])

        pipeline = Pipeline(pipeline_steps)

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
            observers=[RTVIObserver(rtvi)],
        )

        @rtvi.event_handler("on_client_ready")
        async def on_client_ready(rtvi):
            await rtvi.set_bot_ready()
            
            # Signals to the frontend
            messages_rtvi = {
                "show_text_container": True,
                "show_debug_container": True, # Enabled for SOTA analysis
            }
            rtvi_frame = RTVIServerMessageFrame(data=messages_rtvi)
            await task.queue_frames([rtvi_frame])
            
            # Kick off the conversation
            messages.append({"role": "system", "content": "Please introduce yourself to the user."})
            await task.queue_frames([LLMRunFrame()])

        @context_aggregator.user().event_handler("on_bot_turn_started")
        async def on_bot_turn_started(aggregator, strategy):
            logger.info(f"Bot turn started by strategy: {strategy}")

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            # Kick off the conversation
            pass

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await task.queue_frames([EndFrame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
