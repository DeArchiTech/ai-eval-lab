from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import DeepEvalBaseLLM
import anthropic

class ClaudeEval(DeepEvalBaseLLM):
    def __init__(self):
        self.client = anthropic.Anthropic()

    def load_model(self):
        return self.client

    def generate(self, prompt: str) -> str:
        message = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "claude-sonnet-4-6"

claude = ClaudeEval()

test_case = LLMTestCase(
    input="What is Retrieval-Augmented Generation?",
    actual_output="Retrieval-Augmented Generation (RAG) is a technique that combines a retrieval system with a language model. It first retrieves relevant documents from a knowledge base, then uses them as context to generate accurate, grounded responses.",
    retrieval_context=[
        "RAG stands for Retrieval-Augmented Generation. It is an AI framework that retrieves relevant documents from an external knowledge base before generating a response.",
        "RAG reduces hallucinations by grounding the LLM output in retrieved facts rather than relying solely on parametric memory."
    ]
)

answer_relevancy = AnswerRelevancyMetric(threshold=0.7, model=claude)
faithfulness = FaithfulnessMetric(threshold=0.7, model=claude)

evaluate(
    test_cases=[test_case],
    metrics=[answer_relevancy, faithfulness]
)
