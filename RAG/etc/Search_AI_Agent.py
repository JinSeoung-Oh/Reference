### From https://github.com/DhunganaKB/PydanticAIAgent/blob/main/PydanticAI_Agent_1.ipynb

from dotenv import load_dotenv
import asyncio
import os
## For running asyncio
import nest_asyncio
nest_asyncio.apply()
from pydantic_ai import Agent, RunContext
from tavily import TavilyClient, AsyncTavilyClient
import logfire

logfire.configure() # 
#logfire.configure(send_to_logfire='if-token-present')
#'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
load_dotenv() # loading all environment variables

search_agent = Agent(  
    'openai:gpt-4',
    #deps_type=int,
    result_type=str,
    system_prompt=(
        'If the information related to the user question is not availabe, should use the talivy tool'
    ),
)
@search_agent.tool
async def talivy_tool(ctx: RunContext, query:str):  
    """useful to find the latest information from internet"""
    tavily_client = AsyncTavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    response = await tavily_client.search(query, max_results=3)
    return response['results']

# Run the agent
async def run_agent(user_query):
    result = await search_agent.run(user_query)
    return result

# Example usage
user_prompt = "what is gulf of america?"
#user_prompt = "What is the capital city of Nepla?"
response=asyncio.run(run_agent(user_prompt))

--------------------------------------------------------------

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, END, MessageGraph, START
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_openai import ChatOpenAI
from IPython.display import Image, display

# Optional, add tracing in LangSmith
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "Demo-Agent"

tavily_tool = TavilySearchResults(max_results=5)
tools = [tavily_tool]

model = ChatOpenAI(model='gpt-4o')

model_with_tools = model.bind_tools(tools=tools)

builder = MessageGraph()
builder.add_node('model', model_with_tools)
tool_node = ToolNode(tools)
builder.add_node("tool", tool_node)
builder.add_edge(START, "model")

def router(state: list[BaseMessage]):
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "tool"
    else:
        return END

builder.add_conditional_edges("model", router)
builder.add_edge("tool", 'model')

graph = builder.compile()

user_prompt = "what is gulf of america?"
# user_prompt = "who is the president of USA?"
result=graph.invoke(HumanMessage(user_prompt))
print(result)
