import logging
import re
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel

from AIAgentPoC.agents.financialAnalyst.graph import build_graph
from AIAgentPoC.memories.chat_history import memory
from module.observability import (
    configure_logging,
    get_langsmith_client,
    request_config,
    trace_request,
)
from module.settings import get_settings

financialAnalystAgent = build_graph(checkpointer=memory)
logger = logging.getLogger("mt_rag.chat")


@asynccontextmanager
async def lifespan(app):
    configure_logging()
    settings = get_settings()
    logger.info("service_started", extra={"fields": {
        "environment": settings.app_env,
        "revision": settings.app_revision,
        "tracing_enabled": settings.langsmith_tracing,
    }})
    yield
    if settings.langsmith_tracing:
        # The SDK exports in the background. Flush before a normal shutdown.
        import asyncio
        await asyncio.to_thread(get_langsmith_client().flush)


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def stream_reply(message, config):
    # Updates contain only newly produced node output, avoiding repeated history.
    async for update in financialAnalystAgent.astream(
        {"messages": [message]}, config, stream_mode="updates"
    ):
        for node in ("entry_node", "answer_node", "no_related_doc_found_node"):
            for reply in update.get(node, {}).get("messages", []):
                if isinstance(reply, AIMessage) and not reply.tool_calls and reply.text.strip():
                    yield reply.text


@app.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    origin = websocket.headers.get("origin")
    if origin is not None and origin not in get_settings().cors_origins:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    thread_id = str(uuid4())
    logger.info("session_opened", extra={"fields": {"thread_id": thread_id}})
    try:
        while True:
            data = await websocket.receive_text()
            request_id = uuid4()
            config = request_config(thread_id, request_id)
            fields = config["metadata"]
            timeliness = extract_timeliness_number(data)
            if timeliness is not None and timeliness not in range(5):
                await websocket.send_text("时效性评分必须为 0 到 4。")
                continue

            message = HumanMessage(
                content=remove_timeliness_text(data) if timeliness is not None else data,
                additional_kwargs={"timeliness": timeliness} if timeliness is not None else {},
            )
            await websocket.send_text(f"Received: {data}, please wait for a moment...")
            start = perf_counter()
            logger.info("request_started", extra={"fields": fields})
            try:
                # Scope tracing to this request, including the asynchronous graph.
                with trace_request(fields):
                    async for reply in stream_reply(message, config):
                        await websocket.send_text(reply)
                logger.info("request_completed", extra={"fields": {
                    **fields, "duration_ms": round((perf_counter() - start) * 1000, 2),
                }})
            except WebSocketDisconnect:
                raise
            except Exception:
                logger.exception("request_failed", extra={"fields": fields})
                await websocket.send_text(f"处理失败，请重新连接后重试。请求编号: {request_id}")
                await websocket.close(code=1011)
                return
    except WebSocketDisconnect:
        logger.info("session_disconnected", extra={"fields": {"thread_id": thread_id}})
    finally:
        # This demo retains memory only while a connection is alive.
        await memory.adelete_thread(thread_id)


class ChainConfig(BaseModel):
    llm: str
    prompt: str


@app.post("/chain")
async def updateChain(chanConfig: ChainConfig):
    if chanConfig.prompt != "agent_qa":
        return {"response": "Error: Invalid prompt type."}
    return {"response": "Server is connected."}


@app.get("/connect")
async def connect():
    return {"response": "Server is connected."}


@app.get("/health/live")
async def health_live():
    """Process liveness only; never open a database or contact a model."""
    return {"status": "ok"}


def extract_timeliness_number(input_string):
    match = re.search(r"\(timeliness: (\d+)\)", input_string)
    return int(match.group(1)) if match else None


def remove_timeliness_text(input_string):
    return re.sub(r"\(timeliness: \d+\)", "", input_string).strip()
