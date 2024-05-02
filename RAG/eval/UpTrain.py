"""
Response Matching: Assesses the consistency between Answer and Ground Truth.
Response Completeness: Measures whether Answer addresses all aspects of Question.
Response Conciseness: Checks if Answer contains unrelated content.
Response Relevance: Evaluate the relevance between Answer and Question.
Response Validity: Assesses whether Answer is valid, avoiding responses like "I don't know."
Response Consistency: Evaluates consistency between Answer, Question, and Context.
Context Relevance: Measures the relevance between Context and Question.
Context Utilization: Evaluate if Answer utilizes Context to address all points.
Factual Accuracy: Checks if Answer is factually accurate and derived from Context.
Context Conciseness: Measures if Context is concise and avoids irrelevant information.
Context Reranking: Assesses the effectiveness of reranked Context.
Jailbreak Detection: Evaluate whether Question contains jailbreak cues.
Prompt Injection: Measures if Question could lead to leaking system prompts.
Language Features: Assess if Answer is concise, coherent, and free from grammatical errors.
Tonality: Checks if Answer aligns with a specific tone.
Sub-query Completeness: Evaluate if sub-questions cover all aspects of Question.
Multi-query Accuracy: Evaluate if variations of Question align with the original.
Code Hallucination: Measures if code in Answer is relevant to Context.
User Satisfaction: Evaluates user satisfaction in conversations.
"""

import os
import json
from uptrain import EvalLlamaIndex, Evals, ResponseMatching, Settings

settings = Settings(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)
data = []
for i in range(len(questions)):
    data.append(
        {
            "question": questions[i],
            "ground_truth": ground_truth[i],
        }
    )
llamaindex_object = EvalLlamaIndex(settings=settings, query_engine=query_engine)
results = llamaindex_object.evaluate(
    data=data,
    checks=[
        ResponseMatching(),
        Evals.CONTEXT_RELEVANCE,
        Evals.FACTUAL_ACCURACY,
        Evals.RESPONSE_RELEVANCE,
    ],
)
with open("output/uptrain-evaluate.json", "w") as json_file:
    json.dump(results, json_file, indent=2)
