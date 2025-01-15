### From https://towardsdatascience.com/how-to-build-an-ai-agent-for-data-analytics-without-writing-sql-eba811115c1f
### From https://github.com/ChengzhiZhao/AIAgentExamples/tree/main/SQLGenerationAgent

"""
1. Introduction
   The text discusses the development of an AI agent that can handle SQL-related data analytics tasks without requiring users to have SQL expertise.
   By leveraging LangChain and DuckDB, the AI agent is constructed to interpret natural language business inquiries, generate SQL queries, 
   execute them on a database, and then provide comprehensible answers based on the results. 
   A random dataset from Kaggle (Netflix Movies and TV Shows) is used to test the agent’s SQL analysis capabilities.

2. High-Level Workflow for SQL Generation AI Agent
   The agent follows three primary steps:

   -a. write_query:
       -1. Purpose: Convert a user's analytic question into an executable SQL query.
       -2. Example:
           Input: "Can you get the total shows per director and sort by total shows in descending order for the top 3 directors?"
           Output (SQL):

           SELECT director, COUNT(*) as total_shows 
           FROM read_csv_auto('data/netflix_titles.csv') 
           WHERE director IS NOT NULL 
           GROUP BY director 
           ORDER BY total_shows DESC 
           LIMIT 3;
    -b. execute_query:
        -1. Purpose: Execute the generated SQL query using DuckDB to retrieve results.
        -2. Process: Pass the SQL query into DuckDBLoader provided by LangChain to run the query and fetch data.
    -c. generate_answer:
        -1. Purpose: Convert the tabular results from SQL execution into a human-readable answer.
        -2. Process: Feed the question, SQL query, and SQL result back into the LLM to generate a comprehensible summary.
"""
## Step-by-Step Implementation Details
# Step 0: Choose the Right LLM for SQL
# Environment Setup and Configuration:

import os
from langchain_openai import ChatOpenAI
from langchain import hub

# Setup environment variables
os.environ["LANGCHAIN_API_KEY"] = os.environ.get("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.environ.get("LANGCHAIN_TRACING_V2")
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# Set the LLM model
llm = ChatOpenAI(model="gpt-4o")

# Retrieve SQL-specific prompt template from LangChain Hub
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")
----------------------------------------------------------------------------
## Step 1: Build the write_query Function
# Purpose: Generate an SQL query for a given question.
from typing_extensions import Annotated
from typing import TypedDict

class State():
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    query: Annotated[str, ..., "Syntactically valid SQL query."]

def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": "duckdb",
            "top_k": 10,
            "table_info": f"read_csv_auto('{file_path}')",
            "input": state["question"],
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}

# Testing the Function:
sql_query = write_query({"question": "Can you get the total shows per director, and sort by total shows in descending order?"})
print(sql_query)

----------------------------------------------------------------------------
## Step 2: Build the execute_query Function
# Purpose: Execute the SQL query using DuckDB and retrieve the result.
from langchain_community.document_loaders import DuckDBLoader

def execute_query(state: State):
    """Execute SQL query."""
    data = DuckDBLoader(state["query"]).load()
    return {'result': data}

----------------------------------------------------------------------------
## Step 3: Build the generate_answer Function
# Purpose: Summarize the SQL result into a human-readable response.
def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f'Question: {state["question"]}\n'
        f'SQL Query: {state["query"]}\n'
        f'SQL Result: {state["result"]}'
    )
    response = llm.invoke(prompt)
    return {"answer": response.content}

# Combining All Steps
# Integrating Steps into a Workflow Graph:
from langgraph.graph import START, StateGraph

graph_builder = StateGraph(State).add_sequence(
    [write_query, execute_query, generate_answer]
)

graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

"""
3. Monitoring and Further Examples
   -a. Monitoring with LangChain Smith:
       Provides insights into each step's latency, inputs, and outputs within the agent's workflow.
   -b. Additional Query Examples:
       -a. Q1: "Can you get the number of shows that start with letter D?"
           - Expected output includes SQL query generation, execution retrieving the count, and a final human-readable answer.
       -b. Q2: "How many years between each show director Rajiv Chilaka produced, sort by release years?"
           - The system generates a complex SQL query using window functions, executes it, and produces a summarized answer.

4. Final Thought
   The provided text focuses on how to construct an AI agent capable of generating SQL queries, executing them using DuckDB,
   and converting the results into comprehensible answers—all without requiring SQL expertise from the user.
   The approach leverages LangChain for workflow management, DuckDB for query execution, 
   and GPT-4o as the LLM to interpret and generate SQL as well as natural language responses. 
   The detailed code snippets and explanations demonstrate setting up the environment, building each functional component 
   (write_query, execute_query, generate_answer), and orchestrating them into a seamless workflow using LangGraph, 
   aligning strictly with the content provided.
"""
