"""
Bookly AI Customer Support Agent.
Orchestrates conversation with Gemini, tool execution, and response generation.
"""

import json
import logging
import time
from google import genai
from google.genai import types
from mock_data import (
    lookup_order,
    lookup_orders_by_email,
    initiate_return,
    send_password_reset,
    get_policy,
    escalate_to_human,
)

MAX_TOOL_ROUNDS = 5
DEFAULT_MODEL = "gemini-2.5-flash-lite"

logger = logging.getLogger("bookly")

SYSTEM_PROMPT = """
You are Bookly's friendly and professional customer support assistant. Your job is to help 
customers with order inquiries, returns/refunds, and general questions about Bookly, an online bookstore.

## Core Rules

1. **Only answer questions about Bookly.** If a customer asks about something unrelated 
(e.g., medical advice, other companies), politely redirect: "I can only help with Bookly-related questions."
2. **Never invent information.** If you don't have data (e.g., an order isn't found), say so. 
Do not fabricate order statuses, tracking numbers, prices, or policies.
3. **Always use tools to look up data.** Never guess order details, policy specifics, or account info. 
Call the appropriate tool and respond based on what it returns.
4. **Collect required information before acting.** For order lookups, you need an order ID or email. 
For returns, you need the order ID and a reason. For password resets, you need the email. 
Ask for missing info — don't assume.
5. **Confirm destructive actions.** Before initiating a return, confirm the details with the customer first.
6. **Escalate frustrated customers proactively.** If the customer expresses strong frustration, anger, 
or repeats the same request multiple times without resolution, call `escalate_to_human` — don't wait 
for them to ask. It's better to connect them with a human early than to let frustration build.

## What you can do
- Look up order status by order ID or customer email
- Explain shipping, return, password reset, and membership policies
- Initiate returns for eligible orders (delivered within 30 days)
- Send password reset emails
- Escalate to a human agent when the customer requests it
- Answer general questions about Bookly

## What you cannot do
- Modify orders, change shipping addresses, or cancel in-transit shipments
- Access or change passwords directly
- Process refunds outside the return policy
- Provide information about other companies or non-Bookly topics

## Tone
- Friendly and upbeat
- Professional and concise
- Empathetic when customers have problems
"""

# Tool definitions for Gemini's function calling
BOOKLY_TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="lookup_order",
        description="Look up a specific order by its order ID (e.g., ORD-5001). Returns order details including status, items, tracking, and dates.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "order_id": types.Schema(
                    type="STRING",
                    description="The order ID to look up (e.g., ORD-5001)",
                ),
            },
            required=["order_id"],
        ),
    ),
    types.FunctionDeclaration(
        name="lookup_orders_by_email",
        description="Find all orders associated with a customer's email address. Use when customer doesn't have their order ID handy.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "email": types.Schema(
                    type="STRING",
                    description="Customer's email address",
                ),
            },
            required=["email"],
        ),
    ),
    types.FunctionDeclaration(
        name="initiate_return",
        description="Initiate a return for a delivered order. Only call this AFTER confirming the return details with the customer. The order must be delivered and within the 30-day return window.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "order_id": types.Schema(
                    type="STRING",
                    description="The order ID to return",
                ),
                "reason": types.Schema(
                    type="STRING",
                    description="Customer's reason for the return",
                ),
            },
            required=["order_id", "reason"],
        ),
    ),
    types.FunctionDeclaration(
        name="send_password_reset",
        description="Send a password reset email to the customer. Requires their account email address.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "email": types.Schema(
                    type="STRING",
                    description="Customer's email address",
                ),
            },
            required=["email"],
        ),
    ),
    types.FunctionDeclaration(
        name="get_policy",
        description="Retrieve Bookly store policy information. Available topics: 'returns', 'shipping', 'password_reset', 'membership'.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "topic": types.Schema(
                    type="STRING",
                    enum=["returns", "shipping", "password_reset", "membership"],
                    description="The policy topic to look up",
                ),
            },
            required=["topic"],
        ),
    ),
    types.FunctionDeclaration(
        name="escalate_to_human",
        description="Escalate the conversation to a human support agent. Use when the customer explicitly asks to speak to a human or live agent.",
        parameters=types.Schema(
            type="OBJECT",
            properties={
                "reason": types.Schema(
                    type="STRING",
                    description="Why the customer wants to speak to a human (e.g., 'customer requested human agent', 'issue too complex for AI')",
                ),
                "summary": types.Schema(
                    type="STRING",
                    description="Brief summary of the conversation so far to hand off to the human agent",
                ),
            },
            required=["reason", "summary"],
        ),
    ),
])

# Map tool names to handler functions (mock tools for now)
TOOL_HANDLERS = {
    "lookup_order": lambda args: lookup_order(args["order_id"]),
    "lookup_orders_by_email": lambda args: lookup_orders_by_email(args["email"]),
    "initiate_return": lambda args: initiate_return(args["order_id"], args["reason"]),
    "send_password_reset": lambda args: send_password_reset(args["email"]),
    "get_policy": lambda args: get_policy(args["topic"]),
    "escalate_to_human": lambda args: escalate_to_human(args["reason"], args["summary"]),
}

class BooklyAgent:

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        # Create a Gemini client with the provided API key
        self.client = genai.Client(api_key=api_key)
        self.model = model

        # Configure the model with our system prompt and tool definitions
        self.config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            tools=[BOOKLY_TOOLS],
        )

        # Open a chat session — this maintains conversation history across turns
        self.chat_session = self.client.chats.create(
            model=self.model,
            config=self.config,
        )

        # Initialize the event log for debugging and human-agent handoff
        self.events: list[dict] = []
        logger.info(f"BooklyAgent initialized with model: {self.model}")

    def chat(self, user_message: str) -> str:
        """
        Process a user message and return the agent's response.

        Handles the full tool-use loop: sends message to Gemini, executes any
        requested function calls, feeds results back, and repeats until Gemini
        produces a final text response.
        """
        # Send the user's message to Gemini
        self._log_event("user", content=user_message)
        response = self._send_message(user_message)
        if response is None:
            return self._llm_failure_escalation()

        # Agentic loop — if Gemini returns function calls, execute them and
        # send results back. Repeat until Gemini replies with text.
        tool_rounds = 0
        while True:

            # Check for function calls in the response
            parts = response.candidates[0].content.parts or []
            function_calls = [p for p in parts if p.function_call and p.function_call.name]

            # No function calls — return the text response to the customer
            if not function_calls:
                if response.text:
                    self._log_event("assistant", content=response.text)
                    self._log_tokens(response)
                    return response.text
                return self._llm_failure_escalation()

            # Exit 2: Too many tool rounds — escalate to prevent infinite loops
            tool_rounds += 1
            if tool_rounds > MAX_TOOL_ROUNDS:
                logger.error(f"Agentic loop exceeded {MAX_TOOL_ROUNDS} tool rounds — escalating")
                self._log_event("escalation", reason="loop_limit", tool_rounds=tool_rounds)
                esc = escalate_to_human(
                    reason=f"Agentic loop exceeded {MAX_TOOL_ROUNDS} tool rounds",
                    summary="Agent could not resolve the request within the tool round limit.",
                )
                return esc["message"]

            # Log tokens for this Gemini call, then execute the requested tools
            self._log_tokens(response)
            function_responses = []
            for part in function_calls:
                fc = part.function_call
                handler = TOOL_HANDLERS.get(fc.name)

                # Exit 3: Tool failure or unknown tool — escalate without sending error to Gemini
                try:
                    if handler:
                        args = dict(fc.args) if fc.args else {}
                        self._log_event("tool_call", tool=fc.name, args=args)
                        result = handler(args)
                        self._log_event("tool_result", tool=fc.name, data=result)
                    else:
                        raise Exception(f"Unknown tool: {fc.name}")
                except Exception as e:
                    logger.error(f"Tool call failed: {fc.name}(args={fc.args}) — {e}")
                    self._log_event("escalation", reason="tool_failure", tool=fc.name, error=str(e))
                    esc = escalate_to_human(
                        reason=f"Tool '{fc.name}' failed: {e}",
                        summary=f"Customer was mid-conversation when {fc.name} encountered an error.",
                    )
                    return esc["message"]

                # Normalize the tool result into a dict for Gemini's FunctionResponse
                if isinstance(result, dict):
                    response_data = result
                elif isinstance(result, list):
                    response_data = {"results": json.dumps(result, default=str)}
                elif result is None:
                    response_data = {"result": "No data found"}
                else:
                    response_data = {"result": str(result)}

                function_responses.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response=response_data,
                        )
                    )
                )

            # Feed tool results back to Gemini for the next iteration
            response = self._send_message(function_responses)
            if response is None:
                return self._llm_failure_escalation()

    def _log_event(self, role: str, **data):
        """Append a timestamped event to the conversation log."""
        self.events.append({"role": role, "ts": time.time(), **data})

    def _log_tokens(self, response):
        """Log token usage from a Gemini response."""
        usage = response.usage_metadata if response else None
        if usage:
            self._log_event("tokens",
                prompt=usage.prompt_token_count or 0,
                completion=usage.candidates_token_count or 0,
                total=usage.total_token_count or 0,
            )

    def _send_message(self, message) -> object | None:
        """Send a message to Gemini, returning None on API failure."""
        try:
            return self.chat_session.send_message(message)
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            self._log_event("escalation", reason="api_failure", error=str(e))
            return None

    def _llm_failure_escalation(self) -> str:
        """Return a customer-friendly escalation message when the LLM is unavailable."""
        esc = escalate_to_human(
            reason="Gemini API unavailable (rate limit, timeout, or outage)",
            summary="Customer was mid-conversation when the AI service became unavailable.",
        )
        return esc["message"]

    def reset(self):
        """Clear conversation history by starting a new chat session."""
        self.chat_session = self.client.chats.create(
            model=self.model,
            config=self.config,
        )
        self.events = []
