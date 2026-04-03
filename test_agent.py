"""
Unit tests for the Bookly agent's tool execution and escalation logic.
These tests mock the Gemini API so they run without an API key.
"""

from unittest.mock import MagicMock, patch
from agent import BooklyAgent, TOOL_HANDLERS


def make_function_call_response(name, args):
    """Create a mock Gemini response containing a function call."""
    fc = MagicMock()
    fc.function_call.name = name
    fc.function_call.args = args

    response = MagicMock()
    response.candidates = [MagicMock()]
    response.candidates[0].content.parts = [fc]
    response.text = None
    return response


def make_text_response(text):
    """Create a mock Gemini response containing a text reply."""
    part = MagicMock()
    part.function_call = None
    part.text = text

    response = MagicMock()
    response.candidates = [MagicMock()]
    response.candidates[0].content.parts = [part]
    response.text = text
    return response


@patch("agent.genai")
def test_tool_failure_triggers_escalation(mock_genai):
    """When a tool handler throws an exception, the agent should
    immediately escalate to a human instead of sending the error to Gemini."""

    # Mock the client and chat session
    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_chat = MagicMock()
    mock_client.chats.create.return_value = mock_chat

    # Gemini asks to call lookup_order
    mock_chat.send_message.return_value = make_function_call_response(
        "lookup_order", {"order_id": "ORD-5001"}
    )

    agent = BooklyAgent(api_key="fake-key")

    # Temporarily break the handler to simulate a DB outage
    original = TOOL_HANDLERS["lookup_order"]
    TOOL_HANDLERS["lookup_order"] = lambda args: (_ for _ in ()).throw(
        ConnectionError("Database connection timed out")
    )

    try:
        reply = agent.chat("Where is my order ORD-5001?")

        # Should get the escalation message, not a crash
        assert "queue" in reply.lower() or "human" in reply.lower(), (
            f"Expected escalation message, got: {reply}"
        )
        assert "ESC-" in reply or "agent" in reply.lower(), (
            f"Expected ticket or agent reference, got: {reply}"
        )

        # Gemini should only have been called once (the initial send),
        # NOT a second time with an error result
        assert mock_chat.send_message.call_count == 1, (
            f"Expected 1 Gemini call, got {mock_chat.send_message.call_count}. "
            "Tool failure should escalate without going back to Gemini."
        )
    finally:
        TOOL_HANDLERS["lookup_order"] = original


@patch("agent.genai")
def test_successful_tool_call_returns_to_gemini(mock_genai):
    """When a tool succeeds, results should be sent back to Gemini
    for it to formulate a response."""

    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_chat = MagicMock()
    mock_client.chats.create.return_value = mock_chat

    # First call: Gemini asks for lookup_order
    # Second call (with tool results): Gemini responds with text
    mock_chat.send_message.side_effect = [
        make_function_call_response("lookup_order", {"order_id": "ORD-5001"}),
        make_text_response("Your order ORD-5001 has been delivered!"),
    ]

    agent = BooklyAgent(api_key="fake-key")
    reply = agent.chat("Where is my order ORD-5001?")

    assert reply == "Your order ORD-5001 has been delivered!"
    # Two Gemini calls: initial message + tool results
    assert mock_chat.send_message.call_count == 2


@patch("agent.genai")
def test_unknown_tool_triggers_escalation(mock_genai):
    """If Gemini requests a tool that doesn't exist, escalate."""

    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_chat = MagicMock()
    mock_client.chats.create.return_value = mock_chat

    mock_chat.send_message.return_value = make_function_call_response(
        "nonexistent_tool", {"foo": "bar"}
    )

    agent = BooklyAgent(api_key="fake-key")
    reply = agent.chat("Do something weird")

    assert "queue" in reply.lower() or "human" in reply.lower(), (
        f"Expected escalation for unknown tool, got: {reply}"
    )
    assert mock_chat.send_message.call_count == 1


@patch("agent.genai")
def test_max_tool_rounds_triggers_escalation(mock_genai):
    """If the agentic loop exceeds MAX_TOOL_ROUNDS, escalate."""
    from agent import MAX_TOOL_ROUNDS

    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_chat = MagicMock()
    mock_client.chats.create.return_value = mock_chat

    # Gemini keeps requesting tool calls forever (never returns text)
    mock_chat.send_message.return_value = make_function_call_response(
        "get_policy", {"topic": "shipping"}
    )

    agent = BooklyAgent(api_key="fake-key")
    reply = agent.chat("Tell me about shipping")

    assert "queue" in reply.lower() or "human" in reply.lower(), (
        f"Expected escalation after max rounds, got: {reply}"
    )
    # 1 initial call + MAX_TOOL_ROUNDS tool result calls
    assert mock_chat.send_message.call_count == 1 + MAX_TOOL_ROUNDS


@patch("agent.genai")
def test_gemini_api_failure_triggers_escalation(mock_genai):
    """If the Gemini API throws an error (rate limit, timeout, outage),
    the agent should escalate instead of surfacing a raw error."""

    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_chat = MagicMock()
    mock_client.chats.create.return_value = mock_chat

    # Simulate Gemini API failure on the initial message
    mock_chat.send_message.side_effect = Exception(
        "429 RESOURCE_EXHAUSTED: You exceeded your current quota"
    )

    agent = BooklyAgent(api_key="fake-key")
    reply = agent.chat("Where is my order?")

    assert "queue" in reply.lower() or "human" in reply.lower(), (
        f"Expected escalation on API failure, got: {reply}"
    )
    # Should not contain raw error details
    assert "429" not in reply
    assert "RESOURCE_EXHAUSTED" not in reply


@patch("agent.genai")
def test_gemini_api_failure_mid_loop_triggers_escalation(mock_genai):
    """If the Gemini API fails when sending tool results back (mid-loop),
    the agent should escalate gracefully."""

    mock_client = MagicMock()
    mock_genai.Client.return_value = mock_client

    mock_chat = MagicMock()
    mock_client.chats.create.return_value = mock_chat

    # First call succeeds (returns a tool call), second call fails
    mock_chat.send_message.side_effect = [
        make_function_call_response("lookup_order", {"order_id": "ORD-5001"}),
        Exception("503 Service Unavailable"),
    ]

    agent = BooklyAgent(api_key="fake-key")
    reply = agent.chat("Where is my order ORD-5001?")

    assert "queue" in reply.lower() or "human" in reply.lower(), (
        f"Expected escalation on mid-loop API failure, got: {reply}"
    )


