"""
Simple LangGraph Swarm Example
-------------------------------
A swarm of three specialized agents that hand off to each other:

  - Triage Agent   → decides which specialist to use
  - Math Agent     → handles calculations
  - Writing Agent  → handles writing/editing tasks

How it works:
  1. User sends a message to the swarm.
  2. The active agent (starts with Triage) handles the message.
  3. If the task belongs to another agent, it calls a handoff tool.
  4. Control transfers to the specialist; it answers and the loop ends.

Run:
    source .venv/bin/activate
    python swarm_example.py   # uses your AWS SSO credentials automatically
"""

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from langgraph_swarm import create_swarm, create_handoff_tool

# ---------------------------------------------------------------------------
# 1. Shared model  (Claude via Amazon Bedrock)
# ---------------------------------------------------------------------------
model = ChatBedrockConverse(
    model="us.anthropic.claude-sonnet-4-6",
    region_name="us-east-1",
)

# ---------------------------------------------------------------------------
# 2. Handoff tools
#    Each agent gets tools that let it transfer work to the other agents.
# ---------------------------------------------------------------------------

transfer_to_math = create_handoff_tool(
    agent_name="math_agent",
    description="Transfer to the math agent for calculations, equations, or numbers.",
)

transfer_to_writing = create_handoff_tool(
    agent_name="writing_agent",
    description="Transfer to the writing agent for drafting, editing, or creative text.",
)

transfer_to_triage = create_handoff_tool(
    agent_name="triage_agent",
    description="Transfer back to triage if unsure which specialist is needed.",
)

# ---------------------------------------------------------------------------
# 3. Agents
#    Each is a standard ReAct agent with a system prompt + its allowed tools.
# ---------------------------------------------------------------------------

triage_agent = create_agent(
    model=model,
    tools=[transfer_to_math, transfer_to_writing],
    system_prompt=(
        "You are a triage agent. Your job is to route the user's request to the "
        "right specialist. Do NOT answer the question yourself. "
        "For maths/numbers → transfer to math_agent. "
        "For writing/editing/creative → transfer to writing_agent."
    ),
    name="triage_agent",
)

math_agent = create_agent(
    model=model,
    tools=[transfer_to_triage],
    system_prompt=(
        "You are a maths expert. Solve calculations, equations, and number problems "
        "clearly, showing your working. If the request is not about maths, transfer "
        "back to triage."
    ),
    name="math_agent",
)

writing_agent = create_agent(
    model=model,
    tools=[transfer_to_triage],
    system_prompt=(
        "You are a writing expert. Help with drafting, editing, summarising, and "
        "creative writing. If the request is not about writing, transfer back to triage."
    ),
    name="writing_agent",
)

# ---------------------------------------------------------------------------
# 4. Swarm
#    Combines all agents; triage_agent handles every new conversation first.
# ---------------------------------------------------------------------------

swarm = create_swarm(
    agents=[triage_agent, math_agent, writing_agent],
    default_active_agent="triage_agent",
).compile()

# ---------------------------------------------------------------------------
# 5. Run
# ---------------------------------------------------------------------------

def ask(question: str) -> str:
    """Send a question to the swarm and return the final answer."""
    result = swarm.invoke({"messages": [{"role": "user", "content": question}]})
    # The last message in the list is the final assistant reply
    return result["messages"][-1].content


if __name__ == "__main__":
    questions = [
        "What is (17 * 43) + sqrt(256)?",
        "Write a one-sentence tagline for a coffee shop called 'Morning Bloom'.",
    ]

    for q in questions:
        print(f"\nQ: {q}")
        print(f"A: {ask(q)}")
        print("-" * 60)
