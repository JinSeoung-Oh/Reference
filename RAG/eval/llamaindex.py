# From https://generativeai.pub/evaluating-rag-llamaindex-is-all-you-need-852ecd9a3cd3

"""
1. Pros
   No need to install additional third-party libraries, allowing quick usage.
   Evaluation metrics can meet most evaluation needs.

2. Cons
   The evaluation methods primarily rely on LLMs and prompt templates, meaning 
   the evaluation effectiveness can vary significantly depending on the LLM used. 
   Other RAG evaluation tools often use a combination of formulas and prompts to mitigate the LLM’s impact.
   Being a built-in feature of LlamaIndex, it has both advantages and disadvantages. 
   Since evaluation functionality is less critical compared to other RAG functionalities, 
   its development priority may decrease as LlamaIndex introduces more new features.
"""


## Answer Relevance
from llama_index.core.evaluation import AnswerRelevancyEvaluator
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

from llama_index.core.prompts import PromptTemplate

DEFAULT_EVAL_TEMPLATE = PromptTemplate(
    "Your task is to evaluate if the response is relevant to the query.\n"
    "The evaluation should be performed in a step-by-step manner by answering the following questions:\n"
    "1. Does the provided response match the subject matter of the user's query?\n"
    "2. Does the provided response attempt to address the focus or perspective "
    "on the subject matter taken on by the user's query?\n"
    "Each question above is worth 1 point. Provide detailed feedback on response according to the criteria questions above  "
    "After your feedback provide a final result by strictly following this format: '[RESULT] followed by the integer number representing the total score assigned to the response'\n\n"
    "Query: \n {query}\n"
    "Response: \n {response}\n"
    "Feedback:"
)

question = examples[0].query

node_parser = SentenceSplitter()
nodes = node_parser.get_nodes_from_documents(documents)
Settings.llm = llm
vector_index = VectorStoreIndex(nodes)
engine = vector_index.as_query_engine()
response = engine.query(question)
answer = str(response)

print(f"{question}")
print(f"Answer: {answer}")
evaluator = AnswerRelevancyEvaluator(llm)
result = evaluator.evaluate(query=question, response=answer)
print(f"score: {result.score}")
print(f"feedback: {result.feedback}")

# If you need translate 
from llama_index.core.evaluation.answer_relevancy import DEFAULT_EVAL_TEMPLATE

translate_prompt = "\n\nPlease reply in Chinese."
eval_template = DEFAULT_EVAL_TEMPLATE
eval_template.template += translate_prompt
evaluator = AnswerRelevancyEvaluator(
    llm=llm, eval_template=eval_template
)


## Context Relevance
from llama_index.core.evaluation import ContextRelevancyEvaluator

contexts = [n.get_content() for n in response.source_nodes]
evaluator = ContextRelevancyEvaluator(llm)
result = evaluator.evaluate(query=question, contexts=contexts)
print(f"score: {result.score}")
print(f"feedback: {result.feedback}")

## Relevance
from llama_index.core.evaluation import AnswerRelevancyEvaluator

evaluator = RelevancyEvaluator(llm)
result = evaluator.evaluate(query=question, response=answer, contexts=contexts)
print(f"score: {result.score}")
print(f"feedback: {result.feedback}")
print(f"passing: {result.passing}")

## Faithfulness
from llama_index.core.evaluation import FaithfulnessEvaluator

evaluator = FaithfulnessEvaluator(llm)
result = evaluator.evaluate(response=answer, contexts=contexts)
print(f"score: {result.score}")
print(f"feedback: {result.feedback}")
print(f"passing: {result.passing}")

## LlamaIndex’s evaluation tools can assess both retrieval engines and pipelines.
from llama_index.core.query_pipeline import QueryPipeline, InputComponent
from llama_index.core.response_synthesizers.simple_summarize import SimpleSummarize

p = QueryPipeline(verbose=True)
p.add_modules(
    {
        "input": InputComponent(),
        "retriever": retriever,
        "output": SimpleSummarize(),
    }
)
p.add_link("input", "retriever")
p.add_link("input", "output", dest_key="query_str")
p.add_link("retriever", "output", dest_key="nodes")
output = p.run(input=question)
answer = str(output)
contexts = [n.get_content() for n in output.source_nodes]


## Correctness
from llama_index.core.evaluation import CorrectnessEvaluator

evaluator = CorrectnessEvaluator(llm)
ground_truth = dataset_examples[1].reference_answer
print(f"{question}")
print(f"Answer: {answer}")
print(f"Ground Truth: {ground_truth}")
result = evaluator.evaluate(query=question, response=answer, reference=ground_truth)
print(f"score: {result.score}")
print(f"feedback: {result.feedback}")
print(f"passing: {result.passing}")

## Pairwise
documents = SimpleDirectoryReader("./data").load_data()
node_parser = SentenceSplitter(chunk_size=128, chunk_overlap=25)
nodes = node_parser.get_nodes_from_documents(documents)
Settings.llm = llm
vector_index = VectorStoreIndex(nodes)
second_engine = vector_index.as_query_engine()
second_response = engine.query(question)
second_answer = str(second_response)

from llama_index.core.evaluation import PairwiseComparisonEvaluator

print(f"{question}")
print(f"Answer: {answer}")
print(f"Second Answer: {second_answer}")
evaluator = PairwiseComparisonEvaluator(llm)
result = evaluator.evaluate(
    query=question, response=answer, second_response=second_answer
)
print(f"score: {result.score}")
print(f"feedback: {result.feedback}")
print(f"pairwise source: {str(result.pairwise_source)}")

## Batch Evaluation
from llama_index.core.evaluation import BatchEvalRunner

answer_relevancy_evaluator = AnswerRelevancyEvaluator(llm)
context_relevancy_evaluator = ContextRelevancyEvaluator(llm)
relevant_evaluator = RelevancyEvaluator(llm)
correctness_evaluator = CorrectnessEvaluator(llm)
faithfulness_evaluator = FaithfulnessEvaluator(llm)

runner = BatchEvalRunner(
    evaluators={
        "answer_relevancy": answer_relevancy_evaluator,
        "context_relevancy": context_relevancy_evaluator,
        "relevancy": relevant_evaluator,
        "correctness": correctness_evaluator,
        "faithfulness": faithfulness_evaluator,
    },
    workers=8,
)
questions = [example.query for example in examples]
ground_truths = [example.reference_answer for example in examples]
metrics_results = runner.evaluate_queries(
    engine, queries=questions, reference=ground_truths
)

for metrics in metrics_results.keys():
    print(f"metrics: {metrics}")
    eval_results = metrics_results[metrics]
    for eval_result in eval_results:
        print(f"score: {eval_result.score}")
        print(f"feedback: {eval_result.feedback}")
        if eval_result.passing is not None:
            print(f"passing: {eval_result.passing}")








