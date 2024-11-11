## From https://medium.com/@vikram40441/ai-driven-supply-chain-intelligence-adaptive-web-search-with-react-prompting-and-ai-agents-637f2462f83b
## This article provided good insight of AI Agent based system. Have to check given link

import os
from typing import Annotated
from tavily import TavilyClient
from autogen import AssistantAgent, UserProxyAgent, register_function
from autogen.cache import Cache
from autogen.coding import LocalCommandLineCodeExecutor

# Configuration for OpenAI and Tavily API keys
os.environ["OPENAI_API_KEY"] = '<OPEN_API_KEY>'
os.environ["TAVILY_API_KEY"] = '<YOUR_TAVILY_KEY>'

config_list = [{"model": "gpt-3.5-turbo", "api_key": os.environ["OPENAI_API_KEY"]}]
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


# Function to use Tavily API for external searches
def tavily_search_tool(query: Annotated[str, "Supply Chain Risk Query"]) -> Annotated[str, "Search results from Tavily"]:
    return tavily.get_search_context(query=query, search_depth="advanced")

# React Prompts with dynamic, follow-up search instructions
FinancialRisk_prompt = """
Evaluate the financial risk for {supplier_name}.
Paths to consider:
1. Check recent financial performance.
2. Look into debt levels or market reputation.

Use the following format for reasoning:
Thought: Define the specific financial aspect to examine.
Action: Execute Tavily search for that aspect.
Observation: Analyze results.
If debt levels or negative terms (e.g., "bankruptcy", "lawsuit") appear, conduct a follow-up search for recent financial restructuring or lawsuits.
Final Answer: Summarize financial risk findings.

Question: {input}
"""

GeopoliticalRisk_prompt = """
Assess geopolitical risks in {region} affecting {supplier_name}.
Paths to consider:
1. Review political stability and recent events.
2. Examine trade sanctions or regulatory shifts.

Use the following format for reasoning:
Thought: Define the geopolitical aspect to examine.
Action: Execute Tavily search for that aspect.
Observation: Analyze results.
If trade sanctions or political unrest is detected, perform a follow-up search for policy changes and any economic reactions from trading partners.
Final Answer: Summarize geopolitical risk findings.

Question: {input}
"""

EnvironmentalRisk_prompt = """
Assess environmental risks in {location} affecting {supplier_name}.
Paths to consider:
1. Check for recent natural disasters.
2. Examine climate policy changes or environmental regulations.

Use the following format for reasoning:
Thought: Define the environmental aspect to examine.
Action: Execute Tavily search for that aspect.
Observation: Analyze results.
If recurring natural disasters or strict policies are noted, perform a follow-up search on government responses or adaptation incentives.
Final Answer: Summarize environmental risk findings.

Question: {input}
"""

# Define the ReAct prompting messages
def financial_risk_message(sender, recipient, context):
    return FinancialRisk_prompt.format(supplier_name=context["supplier_name"], input=context["question"])

def geopolitical_risk_message(sender, recipient, context):
    return GeopoliticalRisk_prompt.format(region=context["region"], supplier_name=context["supplier_name"], input=context["question"])

def environmental_risk_message(sender, recipient, context):
    return EnvironmentalRisk_prompt.format(location=context["location"], supplier_name=context["supplier_name"], input=context["question"])

# Set up user and assistant agents
user_proxy = UserProxyAgent(
    name="User",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=5,
    code_execution_config={"executor": LocalCommandLineCodeExecutor(work_dir="coding")},
)

financial_agent = AssistantAgent(
    name="FinancialRiskAgent",
    system_message="Assess financial risk using Tavily search and perform follow-up if specific keywords detected. Reply TERMINATE when done.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

geopolitical_agent = AssistantAgent(
    name="GeopoliticalRiskAgent",
    system_message="Assess geopolitical risks using Tavily search with dynamic follow-ups if needed. Reply TERMINATE when done.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

environmental_agent = AssistantAgent(
    name="EnvironmentalRiskAgent",
    system_message="Evaluate environmental risks and perform adaptive search based on Tavily results. Reply TERMINATE when done.",
    llm_config={"config_list": config_list, "cache_seed": None},
)

# Register Tavily search function for each agent
register_function(
    tavily_search_tool,
    caller=financial_agent,
    executor=user_proxy,
    name="tavily_search_tool",
    description="Conducts a search for financial risk data using Tavily",
)

register_function(
    tavily_search_tool,
    caller=geopolitical_agent,
    executor=user_proxy,
    name="tavily_search_tool",
    description="Conducts a search for geopolitical risk data using Tavily",
)

register_function(
    tavily_search_tool,
    caller=environmental_agent,
    executor=user_proxy,
    name="tavily_search_tool",
    description="Conducts a search for environmental risk data using Tavily",
)

# Cache and initiate chat for each agent
with Cache.disk(cache_seed=43) as cache:
    # Financial Risk Assessment
    user_proxy.initiate_chat(
        financial_agent,
        message=financial_risk_message,
        supplier_name="ABC Supplier",
        question="Evaluate the financial risk for ABC Supplier.",
        cache=cache,
    )

    # Geopolitical Risk Assessment
    user_proxy.initiate_chat(
        geopolitical_agent,
        message=geopolitical_risk_message,
        region="Southeast Asia",
        supplier_name="ABC Supplier",
        question="Assess geopolitical risks in Southeast Asia affecting ABC Supplier.",
        cache=cache,
    )

    # Environmental Risk Assessment
    user_proxy.initiate_chat(
        environmental_agent,
        message=environmental_risk_message,
        location="Bangkok, Thailand",
        supplier_name="ABC Supplier",
        question="Assess environmental risks in Bangkok, Thailand affecting ABC Supplier.",
        cache=cache,
    )
