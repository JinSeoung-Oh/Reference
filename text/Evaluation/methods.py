## From https://towardsdatascience.com/llm-evaluation-techniques-and-costs-3147840afc53
## We can use eval for RAG's text generation part for this

## openai
import os
from openai import OpenAI

# Assuming 'openai.api_key' is set elsewhere in the code
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def evaluate_correctness(input, expected_output, actual_output):
    prompt = f"""
    Input: {input}
    Expected Output: {expected_output}
    Actual Output: {actual_output}
    
    Based on the above information, evaluate the correctness of the Actual Output compared to the Expected Output. 
    Provide a score from 0 to 1, where 0 is completely incorrect and 1 is perfectly correct.
    Only return the numerical score.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with evaluating the correctness of outputs.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    return float(response.choices[0].message.content.strip())

if __name__ == "__main__":
    dummy_input = "What is the capital of France?"
    dummy_expected_output = "Paris"
    dummy_actual_output = "Paris corner"

    dummy_score = evaluate_correctness(
        dummy_input, dummy_expected_output, dummy_actual_output
    )
    print(f"Correctness Score: {dummy_score:.2f}")


>> Correctness Score: 0.50

###### Evaluating using a framework
from deepeval.dataset import EvaluationDataset

dataset = EvaluationDataset()
dataset.generate_goldens_from_docs(
    document_paths=['path/to/doc.pdf'],
    max_goldens_per_document=10
)
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

# Initialize the relevancy metric with a threshold value
relevancy_metric = AnswerRelevancyMetric(threshold=0.5)

# Define the test case with input, the LLM's response, and relevant context
test_case = LLMTestCase(
    input="What options do I have if I'm unhappy with my order?",
    actual_output="You can return it within 30 days for a full refund.",
    retrieval_context=["Our policy allows returns within 30 days for a full refund."]
)

# Directly evaluate the test case using the specified metric
assert_test(test_case, [relevancy_metric])


######## Evaluation metrics and RAGs
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

# Define your LLM output and test case
output = "Our working hours are Monday to Friday, 9 AM to 6 PM."
test_case = LLMTestCase(
    input="What are your business hours?", 
    actual_output=output
)

# Initialize the relevancy metric
metric = AnswerRelevancyMetric(threshold=0.7)

# Measure and print the score
metric.measure(test_case)
print(f"Score: {metric.score}, Reason: {metric.reason}")

-------------------------------------------------------------
from deepeval import evaluate
from deepeval.metrics import ContextualPrecisionMetric
from deepeval.test_case import LLMTestCase

# New LLM output and expected response
generated_output = "Our phone support is available 24/7 for premium users."
expected_response = "Premium users have 24/7 access to phone support."

# Contextual information retrieved from RAG pipeline
retrieved_context = [
    "General users don't have phone support", 
    "Premium members can reach our phone support team at any time, day or night.",
    "General users can get email support"
]

# Set up the metric and test case
metric = ContextualPrecisionMetric(threshold=0.8)
test_case = LLMTestCase(
    input="What support options do premium users have?",
    actual_output=generated_output,
    expected_output=expected_response,
    retrieval_context=retrieved_context
)

# Measure and display results
metric.measure(test_case)
print(f"Score: {metric.score}, Reason: {metric.reason}")

------------------------------------------------------------------
from deepeval import evaluate
from deepeval.metrics import ContextualRecallMetric
from deepeval.test_case import LLMTestCase

# New LLM output and expected response
generated_output = "Premium users get access to 24/7 phone support."
expected_response = "Premium users have 24/7 access to phone support."

# Contextual information retrieved from RAG pipeline
retrieved_context = [
    "General users do not have access to phone support.",
    "Premium members can reach our phone support team at any time, day or night.",
    "General users can only get email support."
]

# Set up the recall metric and test case
metric = ContextualRecallMetric(threshold=0.8)
test_case = LLMTestCase(
    input="What support options do premium users have?",
    actual_output=generated_output,
    expected_output=expected_response,
    retrieval_context=retrieved_context
)

# Measure and display results
metric.measure(test_case)
print(f"Recall Score: {metric.score}, Reason: {metric.reason}")

----------------------------------------------------------------
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# New LLM output and corresponding context
actual_output = "Basic plan users can upgrade anytime to the premium plan for additional features."

# Contextual information retrieved from RAG pipeline
retrieved_context = [
    "Users on the Basic plan have the option to upgrade to Premium at any time to gain access to advanced features.",
    "The Premium plan includes additional benefits like 24/7 support and extended storage capacity."
]

# Set up the faithfulness metric and test case
metric = FaithfulnessMetric(threshold=0.75)
test_case = LLMTestCase(
    input="Can Basic plan users upgrade to Premium anytime?",
    actual_output=actual_output,
    retrieval_context=retrieved_context
)

# Measure and display results
metric.measure(test_case)
print(f"Faithfulness Score: {metric.score}, Reason: {metric.reason}")


######## deepval and RAGAS
from deepeval import evaluate
from deepeval.metrics.ragas import RagasMetric
from deepeval.test_case import LLMTestCase

# LLM-generated response, expected response, and retrieved context used to compare model accuracy for a query about product warranty.
llm_response = "The device includes a one-year warranty with free repairs."
target_response = "This product comes with a 12-month warranty and no-cost repairs."
retrieval_context = [
  "All electronic products are backed by a 12-month warranty, including free repair services."
]

# Initialize the Ragas metric with a specific threshold and model configuration
metric = RagasMetric(threshold=0.6)

# Create a test case for the given input and output comparison
test_case = LLMTestCase(
    input="Does this product come with a warranty?",
    actual_output=llm_response,
    expected_output=target_response,
    retrieval_context=retrieval_context
)

# Calculate the metric score for this specific test case
score = metric.measure(test_case)
print(f"Metric Score: {score}")
