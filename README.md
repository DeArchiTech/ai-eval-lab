# AI Eval Lab

Three production-ready AI engineering patterns built from scratch.

## Projects

### 1. LLM Eval Harness (`eval_test.py`)
Custom DeepEval integration using **Claude as the judge** instead of OpenAI.
Runs Answer Relevancy + Faithfulness metrics on RAG outputs.

```bash
pip install deepeval anthropic
python eval_test.py
```

**Why it matters:** Most eval frameworks default to GPT-4. This shows how to swap in any LLM provider — critical for teams with custom model requirements or cost constraints.

---

### 2. LangTree — State Machine Agent (`langtree/langtree.py`)
A 3-node agentic graph built **without LangGraph** — pure Python dataclasses.

```
Question → [Router Node] → Finance Agent OR General Agent → [Summarizer Node]
```

```bash
python langtree/langtree.py
```

**Why it matters:** Understanding *why* LangGraph exists (RL state machines, HJB equations) means you can build the pattern from scratch — not just wrap a library.

---

### 3. Crypto Analyst (`finance/crypto_analyst.py`)
Pulls live SOL/ETH price data from AlphaVantage, calculates 30-day stats, feeds to Claude for plain-English signal generation.

```bash
pip install requests pandas anthropic
python finance/crypto_analyst.py
```

**Output:** Trend, risk level, and Bullish/Bearish/Neutral signal for each coin.

---

## Setup

```bash
python -m venv env
source env/bin/activate
pip install deepeval anthropic requests pandas
export ANTHROPIC_API_KEY=your_key_here
```

---

## Stack
- **Claude** (Haiku + Sonnet) via Anthropic SDK
- **DeepEval** for LLM evaluation metrics
- **AlphaVantage** for financial data
- **Pure Python** state machine (no LangGraph dependency)

---

*Built as part of an AI engineering portfolio. Each project demonstrates understanding of the underlying pattern, not just the wrapper.*
