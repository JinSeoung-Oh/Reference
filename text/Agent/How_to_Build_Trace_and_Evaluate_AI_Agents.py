### From https://medium.com/data-science-collective/how-to-build-trace-and-evaluate-ai-agents-a-python-guide-with-smolagents-and-phoenix-7ee5427b3a1c

# create virtual env
python -m venv env

# Activate 
source env/bin/activate

!pip install smolagents litellm phoenix-evals phoenix-otel openinference-instrumentation-smolagents langchain datasets rank_bm25 python-dotenv tqdm openai

----------------------------------------------------------------------------------------------
### The Retriever Tool (RetrieverTool)

import os
import datasets
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool # Import the base Tool class

# Load API keys from .env file (ensure this is run early)
load_dotenv()

print("Loading and processing knowledge base...")
# Load dataset (consider adding error handling or progress indication)
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# Create LangChain Documents
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# Initialize Text Splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=["\n\n", "\n", ". ", " ", ""], # Adjusted separators slightly
)
docs_processed = text_splitter.split_documents(source_docs)
print(f"Processed {len(source_docs)} documents into {len(docs_processed)} chunks.")

class RetrieverTool(Tool):
    name = "retriever" # How the agent refers to the tool
    description = ( # Description helps the agent decide WHEN to use the tool
        "Uses semantic search (BM25) to retrieve the parts of "
        "Transformers documentation that could be most relevant to answer your query."
    )
    inputs = { # Defines expected input arguments for the LLM
        "query": {
            "type": "string",
            "description": (
                "The query to perform. This should be semantically close to your target "
                "documents. Use the affirmative form rather than a question for better results."
            ),
        }
    }
    output_type = "string" # Defines the expected output type

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        print("Initializing BM25 Retriever...")
        # Initialize the retriever with the processed documents
        self.retriever = BM25Retriever.from_documents(docs, k=10) # Retrieve top 10 chunks
        print("Retriever initialized.")

    def forward(self, query: str) -> str:
        """The actual method called when the agent uses the tool."""
        print(f"RetrieverTool received query: {query}")
        assert isinstance(query, str), "Your search query must be a string"

        # Perform retrieval
        docs = self.retriever.invoke(query)

        # Format the output for the agent
        output = "\nRetrieved documents:\n" + "".join(
            [
                f"\n\n===== Document {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )
        # Optional: print snippet of output for debugging
        # print(f"RetrieverTool returning {len(docs)} documents (first 100 chars): {output[:100]}...")
        return output

# Instantiate the tool with our processed documents
retriever_tool = RetrieverTool(docs=docs_processed)

----------------------------------------------------------------------------------------------------------
### The Search Tool (DuckDuckGoSearchTool)
from smolagents import DuckDuckGoSearchTool

# Instantiate the search tool 
search_tool = DuckDuckGoSearchTool()

----------------------------------------------------------------------------------------------------------
### Setting up the LLM Model
from smolagents import LiteLLMModel

# Make sure OPENAI_API_KEY is loaded from your .env file
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please check your .env file.")

# Configure the model (using gpt-4o-mini as an example)
model = LiteLLMModel(
    model_id="gpt-4o-mini",
    api_key=openai_api_key
)
print(f"Using LLM: {model.model_id}")

-----------------------------------------------------------------------------------------------------------
### Creating and Running the Agent (Untraced)

from smolagents import CodeAgent

# Create the agent instance
manager_agent = CodeAgent(
    tools=[retriever_tool, search_tool], # Pass instances of our tools
    model=model
)

# Initial Test Run (Before Tracing)
print("\n--- Running Agent (Initial Untraced Run) ---")
user_query = "For a transformers model training, which is slower, the forward or the backward pass?"
print(f"User Query: {user_query}")

# Execute the agent run (this might take a moment)
final_answer = manager_agent.run(user_query)

print("\n--- Agent Finished ---")
print(f"Final Answer: {final_answer}")

----------------------------------------------------------------------------------------------------------
### Tracing with Phoenix & OpenTelemetry
## Setting up the Tracer Provider

import phoenix.otel
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

# Define a project name for organizing traces in Phoenix
PROJECT_NAME = "smolagent_evaluation_demo"

print("Initializing Phoenix OpenTelemetry integration...")
tracer_provider = phoenix.otel.register(
    project_name=PROJECT_NAME,
    endpoint="http://127.0.0.1:6006/v1/traces" 
)
print(f"Traces will be sent to Phoenix project: {PROJECT_NAME}")

---------------------------------------------------------------------------------------------------------
### Instrumenting Smolagents
# Automatically instrument Smolagents classes to generate OTel spans
print("Instrumenting Smolagents...")
SmolagentsInstrumentor().instrument()
print("Smolagents instrumentation complete.")

---------------------------------------------------------------------------------------------------------
### Running the Traced Agent
# Running the Agent AGAIN, Now with Tracing Active 
print("\n--- Running Agent (Traced Run) ---")
user_query = "For a transformers model training, which is slower, the forward or the backward pass?"
print(f"User Query: {user_query}")

# Execute the agent run (tracing)
final_answer = manager_agent.run(user_query)

print("\n--- Agent Finished (Traced) ---")
print(f"Final Answer: {final_answer}")
print("\nTrace data should now be available in Phoenix.")

---------------------------------------------------------------------------------------------------------
### Evaluation Task 1: Tool Selection Correctness

import phoenix as px
from phoenix.trace.dsl import SpanQuery
from phoenix.trace import SpanEvaluations
from phoenix.evals import (
    llm_classify,
    OpenAIModel,
)
from openinference.instrumentation import suppress_tracing
import json
import pandas as pd 


print("Querying LLM spans for tool choice evaluation...")
llm_query = SpanQuery().where(
    "span_kind == 'LLM'" # Target the LLM reasoning spans
).select(
    span_id="context.span_id", # Get span_id for logging
    question="input.value",       # The input prompt to the LLM
    generated_code="output.value" # Get the LLM's generated code output
)

llm_decision_df = px.Client().query_spans(llm_query,
                                           project_name=PROJECT_NAME,
                                           timeout=None)

# Filter for spans that actually generated code calling our tools
llm_decision_df = llm_decision_df[
    llm_decision_df['generated_code'].astype(str).str.contains("retriever|web_search", na=False)
].copy()

print(f"Found {len(llm_decision_df)} relevant LLM decision spans.")

# Define the evaluation template (ensure this is defined)
EVAL_GENERATED_CODE_TEMPLATE = """
You are an evaluation assistant. Your task is to determine if the generated Python code correctly uses an available tool to address the user's question.

Available Tools:
- retriever: Uses semantic search on Transformers documentation. Use for documentation-specific questions.
- duckduckgo_search: Searches the web. Use for general knowledge or non-documentation questions.

[BEGIN DATA]
************
[Question]: {question}
************
[Generated Code]: {generated_code}
[END DATA]

Based on the Question, does the Generated Code call the *most appropriate* tool (retriever or duckduckgo_search) with a *reasonable* query argument?

Your response must be a single word, either "correct" or "incorrect".
- "correct" means the most appropriate tool was chosen and the argument seems relevant to the question.
- "incorrect" means the wrong tool was chosen, the argument is unrelated, or the code is malformed/doesn't call a tool.
"""

if not llm_decision_df.empty:
    print("Running tool choice evaluation...")
    with suppress_tracing():
        tool_choice_eval = llm_classify(
            dataframe=llm_decision_df,
            template=EVAL_GENERATED_CODE_TEMPLATE,
            model=OpenAIModel(model="gpt-4o"), # Choose your judge model
            rails=['correct', 'incorrect'],
            provide_explanation=True
        )
    tool_choice_eval['score'] = tool_choice_eval.apply(lambda x: 1 if x['label']=='correct' else 0, axis=1)
    print("Tool choice evaluation complete.")

    # Attach scores back to the original LLM spans in Phoenix
    print("Logging tool choice evaluations...")
    px.Client().log_evaluations(
        SpanEvaluations(eval_name="Tool Choice Eval", dataframe=tool_choice_eval)
    )
    print("Tool choice evaluations logged.")
else:
    print("No relevant LLM decision spans found for tool choice evaluation.")

----------------------------------------------------------------------------------------------------------------
### Evaluation Task 2: Retrieval Relevance
retriever_query = SpanQuery().where(
    "span_kind == 'TOOL' and tool.name == 'retriever'"
).select(
    trace_id="context.trace_id",
    span_id="context.span_id",
    retriever_input="input.value",
    retrieved_docs="output.value",

)

retriever_spans_df = px.Client().query_spans(retriever_query,
                                             project_name=PROJECT_NAME,
                                             timeout=None)

if retriever_spans_df.empty:
    print("Query still returned empty. Double-check project_name and trace data existence.")

retriever_spans_df

EVAL_RETRIEVAL_RELEVANCE_TEMPLATE = """
You are an evaluation assistant. Your task is to assess the relevance of retrieved documents for a given query.

[BEGIN DATA]
************
[Query Sent to Retriever]: {retriever_query}
************
[Retrieved Documents]:
{retrieved_docs}
************
[END DATA]

Based *only* on the [Query Sent to Retriever] and the content of the [Retrieved Documents], are the documents relevant to the query?

- "Relevant": The documents contain information directly related to the topics or entities mentioned in the query.
- "Irrelevant": The documents do not contain information related to the topics or entities mentioned in the query.

Your response must be a single word, either "Relevant" or "Irrelevant".
"""

def extract_query(input_val):
    try:
        input_dict = json.loads(input_val)
        if isinstance(input_dict, dict) and 'args' in input_dict and len(input_dict['args']) > 0:
            return input_dict['args'][0]
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
            if isinstance(input_val, str) and not input_val.strip().startswith('{'):
                return input_val
    return None

if not retriever_spans_df.empty:
    # Create the 'retriever_query' column (index is preserved)
    retriever_spans_df['retriever_query'] = retriever_spans_df['retriever_input'].apply(extract_query)

    # Select only the columns needed for the template, the index ('context.span_id') is kept automatically
    retriever_eval_df = retriever_spans_df[['retriever_query', 'retrieved_docs']].copy()

    # Drop rows where parsing failed or docs are missing
    retriever_eval_df = retriever_eval_df.dropna(subset=['retriever_query', 'retrieved_docs'])

if not retriever_eval_df.empty:
    print("Data prepared for retrieval evaluation (index is context.span_id):")
    print(retriever_eval_df.head())

    print("\nRunning retrieval relevance evaluation...")
    with suppress_tracing():
        retrieval_relevance_eval = llm_classify(
            dataframe=retriever_eval_df,
            template=EVAL_RETRIEVAL_RELEVANCE_TEMPLATE,
            rails=['relevant', 'irrelevant'],
            model=OpenAIModel(model="gpt-4o"),
            provide_explanation=True
        )

    retrieval_relevance_eval['score'] = retrieval_relevance_eval.apply(lambda x: 1 if x['label']=='relevant' else 0, axis=1)

    print("\nRetrieval Relevance Evaluation Results:")
    print(retrieval_relevance_eval.head()) # This DataFrame will also have 'context.span_id' as index

px.Client().log_evaluations(
        SpanEvaluations(eval_name="Retrieval Relevance Eval", dataframe=retrieval_relevance_eval)
    )

