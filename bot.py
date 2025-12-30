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
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import Model, WhisperSTTService
from pipecat.services.xtts.tts import XTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.bot import TurnAnalyzerBotTurnStartStrategy
from pipecat.turns.turn_start_strategies import TurnStartStrategies
from pipecat.runner.types import SmallWebRTCRunnerArguments, RunnerArguments

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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.8)), # Relaxed for Smart Turn
        )
    )

    # 2. Services Setup
    
    # STT: Local Whisper (running on GPU in this container)
    # Using Large V3 Turbo for SOTA punctuation and accuracy on RTX 4090
    stt = WhisperSTTService(model=Model.LARGE_V3_TURBO, device="cuda", no_speech_prob=0.4)

    # LLM: Ollama (Local) running Gemma 3
    # Note: Ensure you have run `ollama pull gemma3:27b`
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "gemma3:27b"), 
    )

    # TTS: XTTS (Local)
    import aiohttp
    async with aiohttp.ClientSession() as session:
        tts = XTTSService(
            voice_id="Claribel Dervla",
            base_url="http://localhost:8000",
            aiohttp_session=session,
        )

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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly intelligent, conversational AI. "
                    "Your responses are synthesized into speech, so adhering to these rules is strict:\n"
                    "1. **Conciseness**: Speak naturally but briefly. Avoid lists and markdown formatting.\n"
                    "2. **Flow**: Do not use 'As an AI' disclaimers. Act with agency.\n"
                    "3. **Vision**: You can see images provided by the user. Describe them naturally if asked.\n"
                    "4. **Personality**: Be helpful, quick-witted, and precise."
                ),
            }
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                turn_start_strategies=TurnStartStrategies(
                    # SOTA: Analyzing the meaning of the turn (complete vs incomplete)
                    bot=[TurnAnalyzerBotTurnStartStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
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
            messages = {
                "show_text_container": True,
                "show_debug_container": True, # Enabled for SOTA analysis
            }
            rtvi_frame = RTVIServerMessageFrame(data=messages)
            await task.queue_frames([rtvi_frame])

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
