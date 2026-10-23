from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect


def turn(websocket, text):
    websocket.send_text(text)
    assert websocket.receive_text().startswith("Received:")
    assert "时效性打分" in websocket.receive_text()
    return websocket.receive_text()


def test_websocket_sessions_and_cleanup(server, fake_rag):
    with TestClient(server.app) as client:
        with client.websocket_connect("/chat") as a, client.websocket_connect("/chat") as b:
            assert turn(a, "A1") == "有据:改写:A1"
            assert turn(b, "B1") == "有据:改写:B1"
            assert turn(a, "A2") == "有据:改写:A2"
    human_histories = [[m.text for m in history if m.type == "human"] for history in fake_rag.entry_inputs]
    assert human_histories == [["A1"], ["B1"], ["A1", "A2"]]
    assert not server.memory.storage


def test_invalid_timeliness_is_reported_without_model_call(server, fake_rag):
    with TestClient(server.app) as client, client.websocket_connect("/chat") as websocket:
        websocket.send_text("问题 (timeliness: 9)")
        assert websocket.receive_text() == "时效性评分必须为 0 到 4。"
        assert not fake_rag.entry_inputs
        assert turn(websocket, "问题 (timeliness: 0)") == "有据:改写:问题"


def test_failure_returns_request_id_and_closes_session(server, monkeypatch):
    class BrokenGraph:
        async def astream(self, *args, **kwargs):
            raise RuntimeError("synthetic backend details")
            yield
    monkeypatch.setattr(server, "financialAnalystAgent", BrokenGraph())
    with TestClient(server.app) as client, client.websocket_connect("/chat") as websocket:
        websocket.send_text("问题")
        assert websocket.receive_text().startswith("Received:")
        error = websocket.receive_text()
        assert "请求编号:" in error
        assert "backend details" not in error
        try:
            websocket.receive_text()
        except WebSocketDisconnect as exception:
            assert exception.code == 1011
        else:
            raise AssertionError("Failed session was not closed")


def test_existing_http_contract(server):
    with TestClient(server.app) as client:
        assert client.get("/connect").json() == {"response": "Server is connected."}
        assert client.post("/chain", json={"llm": "gpt-4o", "prompt": "agent_qa"}).json() == {"response": "Server is connected."}
        assert "Error" in client.post("/chain", json={"llm": "gpt-4o", "prompt": "invalid"}).json()["response"]
