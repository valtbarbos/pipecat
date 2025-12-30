import os
import sys

from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.turn.smart_turn.local_smart_turn_v3 import LocalSmartTurnAnalyzerV3
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import EndFrame, Frame, TranscriptionFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.xtts.tts import XTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.turns.bot import TurnAnalyzerBotTurnStartStrategy
from pipecat.turns.turn_start_strategies import TurnStartStrategies

from pipecat.runner.types import SmallWebRTCRunnerArguments, RunnerArguments

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

class TranscriptionLogger(FrameProcessor):
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, TranscriptionFrame):
            print(f"Transcription: {frame.text}")

        # Push all frames through
        await self.push_frame(frame, direction)

async def bot(runner_args: RunnerArguments):
    webrtc_connection = None
    
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
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=0.2)),
        )
    )

    stt = WhisperSTTService(model="tiny")

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY", "ollama"),
        base_url=os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1"),
        model=os.getenv("LLM_MODEL", "tinyllama"),
    )

    import aiohttp
    async with aiohttp.ClientSession() as session:
        tts = XTTSService(
            voice_id="Claribel Dervla",
            base_url="http://localhost:8000",
            aiohttp_session=session,
        )

        messages = [
            {
                "role": "system",
                "content": "You are a helpful AI assistant. Keep responses very short and concise.",
            }
        ]

        context = LLMContext(messages)
        context_aggregator = LLMContextAggregatorPair(
            context,
            user_params=LLMUserAggregatorParams(
                turn_start_strategies=TurnStartStrategies(
                    bot=[TurnAnalyzerBotTurnStartStrategy(turn_analyzer=LocalSmartTurnAnalyzerV3())]
                ),
            ),
        )
        
        # Transcription logger
        tl = TranscriptionLogger()

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                tl,
                context_aggregator.user(),
                llm,
                tts,
                transport.output(),
                context_aggregator.assistant(),
            ]
        )

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                allow_interruptions=True,
                enable_metrics=True,
            ),
        )

        @transport.event_handler("on_client_connected")
        async def on_client_connected(transport, client):
            # Allow the user to speak first, or just say a fixed hello.
            # Avoid triggering the LLM with "User connected" as it may hallucinate.
            # If we want a greeting, we can inject it.
            # For now, let's keep it silent on connect to prevent the "non-stop talking" issue 
            # allowing the user to say "Hello" and see if STT works.
            pass

        @transport.event_handler("on_client_disconnected")
        async def on_client_disconnected(transport, client):
            await task.queue_frames([EndFrame()])

        runner = PipelineRunner()
        await runner.run(task)


if __name__ == "__main__":
    from pipecat.runner.run import main
    main()
