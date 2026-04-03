"""
API-level tests for the Bookly FastAPI server.
Uses FastAPI's TestClient with a mocked BooklyAgent to test
session management, routing, and error handling.
"""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from server import app, sessions


@patch("server.BooklyAgent")
def test_chat_creates_session(MockAgent):
    """POST /api/chat without a session_id should create a new session."""
    mock_agent = MagicMock()
    mock_agent.chat.return_value = "Hello! How can I help?"
    MockAgent.return_value = mock_agent

    sessions.clear()
    client = TestClient(app)
    resp = client.post("/api/chat", json={"message": "hi"})

    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert data["reply"] == "Hello! How can I help?"
    assert data["session_id"] in sessions


@patch("server.BooklyAgent")
def test_chat_reuses_session(MockAgent):
    """Sending the same session_id should reuse the existing agent."""
    mock_agent = MagicMock()
    mock_agent.chat.side_effect = ["First reply", "Second reply"]
    MockAgent.return_value = mock_agent

    sessions.clear()
    client = TestClient(app)

    r1 = client.post("/api/chat", json={"message": "hello"})
    sid = r1.json()["session_id"]

    r2 = client.post("/api/chat", json={"session_id": sid, "message": "follow up"})
    assert r2.json()["session_id"] == sid

    # Agent constructor should only be called once (session reused)
    assert MockAgent.call_count == 1
    assert mock_agent.chat.call_count == 2


@patch("server.BooklyAgent")
def test_reset_clears_session(MockAgent):
    """POST /api/reset should call agent.reset() for the session."""
    mock_agent = MagicMock()
    mock_agent.chat.return_value = "Hi there"
    MockAgent.return_value = mock_agent

    sessions.clear()
    client = TestClient(app)

    r1 = client.post("/api/chat", json={"message": "hi"})
    sid = r1.json()["session_id"]

    resp = client.post("/api/reset", json={"session_id": sid})
    assert resp.status_code == 200
    mock_agent.reset.assert_called_once()


@patch.dict("os.environ", {"GOOGLE_API_KEY": ""}, clear=False)
def test_chat_missing_api_key():
    """Should return 500 when GOOGLE_API_KEY is not set."""
    sessions.clear()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 500


def test_timing_header():
    """Every response should include the X-Process-Time header."""
    client = TestClient(app)
    resp = client.get("/")
    assert "x-process-time" in resp.headers
    assert resp.headers["x-process-time"].endswith("s")


@patch("server.BooklyAgent")
def test_agent_exception_returns_500(MockAgent):
    """If agent.chat() raises, the server should return a 500."""
    mock_agent = MagicMock()
    mock_agent.chat.side_effect = RuntimeError("unexpected failure")
    MockAgent.return_value = mock_agent

    sessions.clear()
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.post("/api/chat", json={"message": "hi"})
    assert resp.status_code == 500
