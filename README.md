# Bookly AI Customer Support Agent

A conversational AI agent for **Bookly**, a fictional online bookstore. Built with Google Gemini, FastAPI, and a minimal web chat UI.

## Quick Start

### Prerequisites
- Python 3.11+
- A free [Google AI Studio API key](https://aistudio.google.com/apikey)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your API key or use local .env file
export GOOGLE_API_KEY="your-api-key-here"

# Start the server
python3 -m uvicorn server:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

### Try these scenarios

| Scenario | Example message |
|----------|----------------|
| Order status | "Where's my order ORD-5001?" |
| Return flow (multi-turn) | "I want to return a book" |
| Policy question | "What's your shipping policy?" |
| Password reset | "I forgot my password" |
| Email-based lookup | "Can you look up orders for alice@example.com?" |
| Human escalation | "I want to speak to a real person" |
| Out-of-scope | "What's the weather today?" |

### Test data

| Order ID | Customer | Status | Returnable? |
|----------|----------|--------|-------------|
| ORD-5001 | alice@example.com | Delivered (3 days ago) | Yes |
| ORD-5002 | alice@example.com | Shipped | No (not delivered) |
| ORD-5003 | bob@example.com | Processing | No (not delivered) |
| ORD-5004 | carol@example.com | Delivered (38 days ago) | No (past 30-day window) |

## Project Structure

```
├── server.py          # FastAPI server & API routes
├── agent.py           # Core agent: system prompt, tools, agentic loop
├── mock_data.py       # Simulated orders, customers, policies
├── test_agent.py      # Unit tests (run: pytest test_agent.py -v)
├── test_server.py     # API tests — session management, routing, errors
├── eval_harness.py    # Eval suite — 9 scenarios against live Gemini API
├── static/
│   └── index.html     # Web chat interface + debug panel
├── DESIGN.md          # Architecture & design decisions
└── README.md          # This file
```

## Design Document

See [DESIGN.md](DESIGN.md) for architecture overview, conversation design, hallucination controls, and production readiness discussion.
