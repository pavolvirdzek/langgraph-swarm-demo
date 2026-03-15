# LangGraph Swarm Demo

A minimal example of a multi-agent swarm using [langgraph-swarm](https://github.com/langchain-ai/langgraph-swarm) and Claude via Amazon Bedrock.

## What it does

Three specialized agents collaborate by handing off work to each other:

- **Triage agent** — receives every message and routes it to the right specialist
- **Math agent** — handles calculations and equations
- **Writing agent** — handles drafting, editing, and creative text

```
User: "What is (17 * 43) + sqrt(256)?"

triage_agent  →  "this is maths"  →  transfer_to_math
math_agent    →  solves it        →  "747"
```

## Setup

**Requirements:** Python 3.10+, AWS credentials with Bedrock access

```bash
python -m venv .venv
source .venv/bin/activate
pip install langchain-anthropic langgraph langgraph-swarm langchain-aws
```

## Run

```bash
source .venv/bin/activate
python swarm_example.py
```

AWS credentials are picked up automatically from your environment (`~/.aws/credentials`, SSO, IAM role, etc.).

## How it works

Each agent is a ReAct loop — it reasons, optionally calls a tool, and repeats until it gives a final answer. The only tools available are **handoff tools**: they don't do any computation, they just transfer control (and the full message history) to another agent.

`create_swarm` wires all agents into a LangGraph state machine. The active agent is tracked in the graph state and changes on each handoff.

## Cost note

Each handoff is a separate API call with the full message history re-sent each time — expect 2–3× more tokens than a single-agent approach. The tradeoff is better specialisation and a cleaner separation of concerns.
