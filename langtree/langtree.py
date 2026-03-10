import anthropic
from dataclasses import dataclass, field
from typing import Literal

client = anthropic.Anthropic()

# --- State ---
@dataclass
class State:
    question: str
    route: Literal["finance", "general"] = None
    answer: str = None
    summary: str = None

# --- Nodes ---
def node_a_router(state: State) -> State:
    """Route the question to finance or general agent"""
    prompt = f"""Classify this question as either 'finance' or 'general'.
Question: {state.question}
Reply with only one word: finance or general"""

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": prompt}]
    )
    state.route = response.content[0].text.strip().lower()
    print(f"[Node A] Routed to: {state.route}")
    return state

def node_b_finance(state: State) -> State:
    """Finance specialist agent"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": f"You are a financial analyst. Answer concisely: {state.question}"}]
    )
    state.answer = response.content[0].text
    print(f"[Node B] Finance answer generated")
    return state

def node_c_general(state: State) -> State:
    """General knowledge agent"""
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        messages=[{"role": "user", "content": f"Answer concisely: {state.question}"}]
    )
    state.answer = response.content[0].text
    print(f"[Node C] General answer generated")
    return state

def node_summarizer(state: State) -> State:
    """Final node - summarize in one sentence"""
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": f"Summarize in one sentence: {state.answer}"}]
    )
    state.summary = response.content[0].text
    print(f"[Summarizer] Done")
    return state

# --- Graph transitions ---
def run(question: str) -> State:
    state = State(question=question)
    state = node_a_router(state)

    # Conditional transition
    if state.route == "finance":
        state = node_b_finance(state)
    else:
        state = node_c_general(state)

    state = node_summarizer(state)
    return state

# --- Run ---
if __name__ == "__main__":
    questions = [
        "What is the current trend in S&P 500 valuations?",
        "What is the speed of light?"
    ]
    for q in questions:
        print(f"\n{'='*50}")
        print(f"Question: {q}")
        result = run(q)
        print(f"Summary: {result.summary}")
