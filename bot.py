import os
import sys

from loguru import logger
from dotenv import load_dotenv

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMMessagesFrame, EndFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.processors.aggregators.llm_response import LLMUserResponseAggregator
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.services.xtts.tts import XTTSService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport

from pipecat.runner.types import SmallWebRTCRunnerArguments, RunnerArguments

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

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
            vad_analyzer=SileroVADAnalyzer(),
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

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMUserResponseAggregator(messages)

        pipeline = Pipeline(
            [
                transport.input(),
                stt,
                tma_in,
                llm,
                tma_out,
                tts,
                transport.output(),
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
