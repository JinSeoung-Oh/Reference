## From https://ai.gopubby.com/adding-memory-to-agents-in-llm-based-production-ready-applications-9274f7381369

## After setting env

import dotenv
%load_ext dotenv
%dotenv

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.sqlite import SqliteSaver

tool = TavilySearchResults(max_results=2)

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

model = ChatOpenAI(model="gpt-4")
tools = [tool]
model_with_tools = model.bind_tools(tools)
agent_executor = create_react_agent(model, tools)

memory = SqliteSaver.from_conn_string("sqlite.sqlite")
agent_executor = create_react_agent(model, tools, checkpointer=memory)

config = {"configurable": {"thread_id": "test_thread_sqlite"}}

for chunk in agent_executor.stream(
    {"messages": [HumanMessage(content="Who is Thomas to John")]}, config
):
    print(chunk)

print(chunk["agent"]["messages"][0].content)
