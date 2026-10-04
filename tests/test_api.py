from fastapi.testclient import TestClient

from app.ai.agent import AgentResult
from app.ai.sessions import session_store
from app.main import app
from app.services import chat as chat_service


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_local_frontend_origin_is_allowed_by_cors() -> None:
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"


def test_unlisted_origin_is_not_allowed_by_cors() -> None:
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "https://untrusted.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_chat_rejects_blank_message() -> None:
    response = client.post("/api/chat", json={"message": "   "})
    assert response.status_code == 422


def test_debug_routes_do_not_accept_identity_parameter() -> None:
    openapi = client.get("/openapi.json").json()
    for path in (
        "/api/debug/resourceplus/attendance",
        "/api/debug/resourceplus/home",
        "/api/debug/resourceplus/profile",
    ):
        parameters = openapi["paths"][path]["get"].get("parameters", [])
        assert "usr_email" not in {parameter["name"] for parameter in parameters}


def test_chat_generates_and_returns_session_id(monkeypatch) -> None:
    session_store.clear()

    async def run_agent(message, *, lang, session_id, history, response_language):
        assert message == "Hello"
        assert history == []
        assert response_language == "en"
        return AgentResult(message="Hello!", tools_used=[])

    monkeypatch.setattr(chat_service, "run_agent", run_agent)
    response = client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 200
    assert response.json()["message"] == "Hi! What can I help you with?"
    assert response.json()["session_id"]
    assert response.json()["requires_confirmation"] is False


def test_chat_returns_safe_live_reason_options_without_ids(monkeypatch) -> None:
    session_store.clear()

    async def run_agent(message, *, lang, session_id, history, response_language):
        return AgentResult(
            message="What was the reason?",
            tools_used=["prepare_less_hours_correction"],
            needs_reason=True,
            reason_options=["Embassy Purposes", "Family Circumstances"],
        )

    monkeypatch.setattr(chat_service, "run_agent", run_agent)
    response = client.post(
        "/api/chat",
        json={"message": "Please help me with this HR item"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "What was the reason?"
    assert body["display_message"] == "What was the reason?"
    assert body["needs_reason"] is True
    assert body["requires_confirmation"] is False
    assert body["reason_options"] == [
        {"label": "Embassy Purposes", "value": "Embassy Purposes"},
        {"label": "Family Circumstances", "value": "Family Circumstances"},
    ]
    assert "reasonID" not in response.text
    assert "reason_id" not in response.text
