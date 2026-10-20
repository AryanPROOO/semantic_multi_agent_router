import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from main import process_user_query


def test_routing_accuracy():
    """Verify Qdrant routes each domain query to the exact intended SLM agent."""
    test_cases = [
        ("Implement binary search in Python", "coding_agent"),
        ("Find the derivative of cos(x)", "math_agent"),
        ("What is the enterprise SLA uptime guarantee?", "factual_rag_agent"),
    ]
    for query, expected_agent in test_cases:
        output = process_user_query(query)
        assert output["routed_to"] == expected_agent, f"Failed for {query}"


def test_factual_consistency_and_relevancy():
    """Score the Factual RAG Agent response for Faithfulness (factual consistency) and Relevancy."""
    query = "What is the refund policy and what is the enterprise tier rate limit?"
    result = process_user_query(query)
    test_case = LLMTestCase(
        input=query,
        actual_output=result["response"],
        retrieval_context=result["retrieval_context"],
    )
    # FaithfulnessMetric checks if the response hallucinates outside retrieval_context
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, include_reason=True)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, include_reason=True)
    assert_test(test_case, [faithfulness_metric, relevancy_metric])
