"""Interactive demo; running this calls OpenAI."""

import asyncio
import io
from pathlib import Path
import sys
from uuid import uuid4

sys.path.append(str(Path(__file__).parents[3]))

from langchain_core.messages import HumanMessage
from PIL import Image
from AIAgentPoC.agents.timelinessGrader.graph import graph
from module.observability import configure_logging, get_langsmith_client, request_config, trace_request
from module.settings import get_settings


async def main():
    configure_logging()
    png_data = graph.get_graph().draw_mermaid_png()
    Image.open(io.BytesIO(png_data)).save(Path(__file__).with_name("graph.png"))
    config = request_config(str(uuid4()), uuid4())
    try:
        with trace_request(config["metadata"]):
            output = await graph.ainvoke(
                {"messages": [HumanMessage(content="请教“三巫日”真的会导致股市波动吗？")]},
                config,
            )
        score = output["final_response"]
        print("score:", score.value)
        print("reason:", score.reason)
    finally:
        if get_settings().langsmith_tracing:
            await asyncio.to_thread(get_langsmith_client().flush)


if __name__ == "__main__":
    asyncio.run(main())
