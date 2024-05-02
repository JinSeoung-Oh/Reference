"""
Faithfulness: Evaluates consistency between Question and Context.
Answer Relevance: Assesses consistency between Answer and Question.
Contextual Precision: Checks whether Ground Truth ranks high in Context.
Contextual Recall: Evaluates consistency between Ground Truth and Context.
Contextual Relevancy: Evaluates consistency between Question and Context.
Hallucination: Measures the degree of hallucinations.
Bias: Evaluates bias levels.
Toxicity: Measures the presence of toxicity, including personal attacks, sarcasm, or threats.
Ragas: Allows for using Ragas for evaluation and generating explanations.
Knowledge Retention: Evaluates the persistence of information.
Summarization: Evaluate the effectiveness of summarization.
G-Eval: G-Eval is a framework for performing evaluation tasks using a Large Language Model (LLM) with Chain of Thought (CoT). It can evaluate LLM outputs based on any custom criteria. For more information, check out this paper.
"""

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRelevancyMetric,
)
from deepeval.test_case import LLMTestCase
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset

def generate_dataset():
    test_cases = []
    for i in range(len(questions)):
        response = query_engine.query(questions[i])
        test_case = LLMTestCase(
            input=questions[i],
            actual_output=response.response,
            retrieval_context=[node.get_content() for node in response.source_nodes],
            expected_output=ground_truth[i],
        )
        test_cases.append(test_case)
    return EvaluationDataset(test_cases=test_cases)

dataset = generate_dataset()

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_rag(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(model="gpt-3.5-turbo")
    faithfulness_metric = FaithfulnessMetric(model="gpt-3.5-turbo")
    context_relevancy_metric = ContextualRelevancyMetric(model="gpt-3.5-turbo")
    assert_test(
        test_case,
        [answer_relevancy_metric, faithfulness_metric, context_relevancy_metric],
    )
