# Bookly AI Support Agent — Design Document

## Architecture Overview

The agent follows a **tool-augmented LLM** pattern with an agentic loop:

```
                                          ┌─────────────────┐
                                          │   Gemini API    │
                                          │ (function calls)│
                                          └────────┬────────┘
                                                   │
                                          API error? ──YES──┐
                                                   │        │
                                                   NO       │
                                                   ▼        │
User ──► FastAPI ──► BooklyAgent ◄──► Agentic Loop          │
                                        │      │            │
                                        │   rounds > 5? ───►│
                                        │      │            │
                                        ▼      ▼            ▼
                                  Tool Execution      ┌───────────┐
                                  Layer               │ Escalate  │
                                   │                  │ to Human  │
                              exception? ──YES──────► └───────────┘
                                   │
                                   NO
                                   ▼
                              Mock Data Store
```

**Components:**

| Component | Role |
|-----------|------|
| `server.py` | FastAPI app — routes HTTP requests, manages sessions, loads `.env` config |
| `agent.py` | Core agent — system prompt, 6 tool definitions, agentic loop with safety guards |
| `mock_data.py` | Simulated database — orders, customers, policies, escalation |
| `test_agent.py` | Unit tests — mocks Gemini API to verify tool execution, escalation, and loop limits |
| `test_server.py` | API tests — mocks BooklyAgent to verify session management, routing, and error handling |
| `eval_harness.py` | Evaluation suite — runs 9 conversation scenarios against the live agent, asserts on event trace |
| `static/index.html` | Single-page web chat UI with debug panel for inspecting the agent's event trace |

**Session management:** Each browser session gets a unique ID mapped to a `BooklyAgent` instance with its own conversation history. This ensures multi-turn context is preserved.

**Agentic loop:** When the user sends a message, the agent calls Gemini, which may respond with text *or* function calls. If Gemini requests function calls, the agent executes them locally, feeds results back, and repeats — allowing Gemini to chain multiple calls (e.g., look up an order, then check return eligibility) before producing a final text response. The loop has four exit conditions:

1. **Normal exit** — Gemini responds with text (no function calls)
2. **Sentiment-driven escalation** — Gemini detects customer frustration or an explicit request to speak to a human and calls `escalate_to_human` (LLM-driven, via system prompt)
3. **Tool failure** — a handler throws an exception → auto-escalate to human (deterministic, bypasses Gemini)
4. **Loop limit** — more than `MAX_TOOL_ROUNDS` (5) iterations → auto-escalate to human (deterministic, bypasses Gemini)

## Tools

| Tool | Purpose | Sample Trigger |
|------|---------|----------|
| `lookup_order` | Retrieve order details by ID | "Where's my order ORD-5001?" |
| `lookup_orders_by_email` | Find all orders for a customer email | "Can you look up orders for alice@example.com?" |
| `initiate_return` | Process a return (after confirmation) | "Yes, please return it" |
| `send_password_reset` | Trigger a password reset email | "I forgot my password" |
| `get_policy` | Retrieve store policies (shipping, returns, membership, password reset) | "What's your return policy?" |
| `escalate_to_human` | Hand off to a human agent with conversation summary | Customer request, frustration, or system failure |

## Conversation & Decision Design

The system prompt establishes four decision modes:

1. **Answer directly** — for policy questions and greetings. The model calls `get_policy` for factual accuracy, then summarizes in natural language.
2. **Ask a clarifying question** — when required information is missing. For example, "I want to return a book" triggers the agent to ask for the order ID before proceeding. The system prompt explicitly instructs: "Ask for missing info — don't assume."
3. **Take an action** — for returns and password resets. The agent must confirm details with the customer before calling destructive tools like `initiate_return`.
4. **Escalate to a human** — when the customer explicitly requests it, expresses strong frustration, or when the system detects a failure condition.

**Multi-turn example (return flow):**
1. Customer: "I want to return a book"
2. Agent: "I'd be happy to help with a return. Could you provide your order ID?"
3. Customer: "ORD-5001"
4. Agent: *(calls `lookup_order`)* "I see order ORD-5001 with The Great Gatsby and 1984, delivered on March 29. Which book would you like to return, and what's the reason?"
5. Customer: "Wrong edition of Gatsby"
6. Agent: "I'll initiate a return for The Great Gatsby from ORD-5001 due to wrong edition. Shall I proceed?"
7. Customer: "Yes"
8. Agent: *(calls `initiate_return`)* "Done! Return RET-5001 has been created. A prepaid return label has been emailed..."

**Why this design:** Structuring the prompt around explicit decision modes — rather than leaving the LLM to figure out when to act — reduces unpredictable behavior. The confirmation step before destructive actions is a deliberate guardrail, not just politeness.

## Escalation Design

Escalation to a human agent happens through three pathways — one LLM-driven, two deterministic:

| Pathway | Trigger | Who decides |
|---------|---------|-------------|
| **Customer request** | "I want to speak to a person" | Gemini (calls `escalate_to_human` tool) |
| **Frustration detection** | Angry tone, repeated unresolved requests | Gemini (system prompt instructs proactive escalation) |
| **Tool failure** | Handler throws an exception (e.g., DB timeout) | Code (try/except in agentic loop, bypasses Gemini) |
| **Loop limit exceeded** | More than 5 tool round-trips in one turn | Code (counter in agentic loop, bypasses Gemini) |

**Why two layers:** The LLM handles nuanced signals (frustration, explicit requests) where language understanding matters. The code handles mechanical failures (exceptions, infinite loops) where deterministic behavior is essential. Tool failures and loop limits bypass Gemini entirely — no risk of the LLM misinterpreting an error or getting stuck in a retry loop.

## Hallucination & Safety Controls

| Control | How it works |
|---------|-------------|
| **Tool-gated facts** | The system prompt says "Never guess order details...call the appropriate tool." The model cannot respond with order/policy data without first retrieving it via a function call. |
| **Domain boundary** | The prompt explicitly restricts scope to Bookly topics and lists what the agent *cannot* do (modify orders, access passwords). This prevents confident-sounding answers about out-of-scope topics. |
| **Structured tool outputs** | Tools return typed JSON, not free text. The model interprets structured data rather than hallucinating from a vague context. |
| **Input validation** | Tools validate order ID format (`ORD-XXXX`) and email format before processing. Invalid inputs return descriptive errors instead of silent "not found" responses, giving the model useful signal to relay to the customer. |
| **No-data honesty** | When a tool returns null/empty (e.g., order not found), the system prompt instructs the model to say so rather than fabricate. |
| **Confirmation before action** | Returns require explicit customer confirmation, preventing unintended irreversible actions. |
| **Loop limit** | The agentic loop is capped at 5 tool rounds per turn, preventing infinite loops or runaway API costs. |

## Production Readiness — Tradeoffs & Next Steps

**Tradeoffs made for speed:**

- **In-memory sessions** — sessions are lost on restart. Production would use Redis or a database with TTL-based expiration.
- **Synchronous LLM calls** — the endpoint blocks while waiting for Gemini. Production would use async streaming for better UX and server throughput.
- **No authentication** — any request can create a session. Production would tie sessions to authenticated customer accounts if it can.
- **Mock data** — all data is hardcoded. Production would integrate with real order management, CRM, and auth systems via API calls.
- **Static escalation** — `escalate_to_human` returns a mock response. Production would create a real support ticket, enter a queue, and hand off conversation context.
- **Observability** — the agent logs a per-session event trace (user messages, tool calls, tool results, escalations, token usage, and assistant responses with timestamps). The server emits an `X-Process-Time` header on every response. A debug panel in the chat UI visualizes the event trace in real time, showing color-coded events, elapsed timing, and token counts per Gemini call. Production would add persistent storage for event traces, structured logging with session IDs, and dashboards for latency and tool failure rates.

**What I'd change for production:**

1. **Streaming responses** — use Gemini's streaming API and SSE to show tokens as they arrive, reducing perceived latency.
2. **Conversation persistence** — store conversation history in a database for audit, handoff to human agents, and analytics.
3. **Observability** — log every function call, LLM request/response, and latency. Tag with session ID for tracing. Track call frequency, failure rates, and response times.
4. **Real escalation pipeline** — integrate with a ticketing system and pass full conversation context so the human agent doesn't start from scratch.
5. **Guardrail layer** — add input/output classifiers to catch prompt injection, PII leakage, and off-topic responses as a second defense beyond the system prompt.
6. **Rate limiting & auth** — protect the API from abuse and tie sessions to authenticated customer identities.
7. **Customer identity resolution** — in production, the agent should know who the customer is from their login session, eliminating the need to ask for user input in many cases.

## Metrics & Rating

The agent tracks both subjective and objective quality signals:

**Customer rating** — A 5-star rating appears below the latest assistant message in the chat UI. Clicking submits via `POST /api/rating` and records a `rating` event in the session's event trace. The rating persists across turns so the customer can update it as the conversation progresses.

**Aggregate metrics** — `GET /api/stats` computes across all active sessions:

| Metric | How it's computed |
|--------|-------------------|
| `deflection_rate` | % of sessions that resolved without any escalation |
| `resolution_rate` | % of sessions where an action tool (`initiate_return`, `send_password_reset`) succeeded without escalation |
| `avg_rating` | Mean of all submitted star ratings |
| `avg_turns` | Average customer messages per session |

Deflection and resolution are derived automatically from the event trace — no customer input needed. The star rating captures subjective satisfaction that event data alone can't measure.

## Evaluation Suite

`eval_harness.py` runs 9 conversation scenarios against the live Gemini API and asserts on the **event trace** — checking which tools were called, whether escalation occurred, and whether replies contain expected content. This tests LLM behavior, not just plumbing.

| Scenario | What it tests |
|----------|--------------|
| Order lookup by ID | `lookup_order` called, reply mentions "delivered" |
| Order lookup by email | `lookup_orders_by_email` called |
| Policy question | `get_policy` called, reply mentions "shipping" |
| Return flow (multi-turn) | `initiate_return` called across 3-turn confirmation flow |
| Return rejected | Past 30-day window, reply mentions "30" |
| Password reset | `send_password_reset` called |
| Human escalation | `escalate_to_human` called on explicit request |
| Out-of-scope rejection | No tools called, reply mentions "Bookly" |
| Prompt injection | Agent stays in scope, doesn't mention "flight" or "Paris" |

Run with: `GOOGLE_API_KEY=your-key python eval_harness.py`

## System Prompt

See `agent.py:SYSTEM_PROMPT` for the full prompt. Key design choices in the prompt:

- **Domain scoping** — "Only answer questions about Bookly" with a polite redirect pattern for off-topic requests (e.g., medical advice, other companies)
- **Explicit capability boundaries** ("What you can do" / "What you cannot do") to prevent scope creep
- **Tool-use mandate** — "Always use tools to look up data" prevents hallucinated facts
- **Action confirmation** — "Confirm destructive actions" prevents accidental returns
- **Information-first** — "Collect required information before acting" drives the multi-turn clarification behavior
- **Proactive escalation** — "Escalate frustrated customers proactively" uses the LLM's language understanding for sentiment-based handoff
- **Tone directives** — friendly, professional, and empathetic. Explicit tone guidance ensures consistent voice across conversations and prevents overly formal or robotic responses.
