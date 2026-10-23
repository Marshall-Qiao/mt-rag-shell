"""Interactive demo; running this calls OpenAI and reads the configured vector DB."""

import asyncio
import io
from pathlib import Path
import sys
from uuid import uuid4

sys.path.append(str(Path(__file__).parents[3]))

from langchain_core.messages import HumanMessage
from PIL import Image
from AIAgentPoC.agents.financialAnalyst.graph import build_graph
from AIAgentPoC.memories.chat_history import memory
from module.observability import configure_logging, get_langsmith_client, request_config, trace_request
from module.settings import get_settings


async def main():
    configure_logging()
    graph = build_graph(checkpointer=memory)
    png_data = graph.get_graph().draw_mermaid_png()
    Image.open(io.BytesIO(png_data)).save(Path(__file__).with_name("graph.png"))
    config = request_config(str(uuid4()), uuid4())
    try:
        with trace_request(config["metadata"]):
            async for output in graph.astream(
                {"messages": [HumanMessage(content="特斯拉最近经营状况怎么样？")]},
                config,
                stream_mode="values",
            ):
                if "messages" in output:
                    output["messages"][-1].pretty_print()
    finally:
        await memory.adelete_thread(config["configurable"]["thread_id"])
        if get_settings().langsmith_tracing:
            await asyncio.to_thread(get_langsmith_client().flush)


if __name__ == "__main__":
    asyncio.run(main())
