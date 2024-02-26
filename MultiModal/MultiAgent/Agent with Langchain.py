##From https://heartbeat.comet.ml/implementing-agents-in-langchain-a26e1e737c31


# In LangChain, two fundamental concepts are used: agents and chains. 
# Agents use a Language and Logic Model (LLM) to determine a sequence of actions, while chains are hardcoded sequences of actions.
# The main distinction is that agents focus on decision-making and interaction with the environment, whereas chains focus on the flow of information and computation.

# Agents use an LLM as a reasoning engine and connect it to two primary components: 
# tools and memory. Tools and toolkits provide additional functionality to agents, 
# enabling them to perform tasks like accessing external resources, processing and manipulating data, 
# integrating with other systems, and customization. Memory is crucial for agents to store and retrieve information during decision-making, 
# allowing them to maintain context, accumulate knowledge, personalize responses, and ensure continuity in conversations or interactions.

## Setup LLM
%%capture
!pip install langchain openai duckduckgo-search youtube_search wikipedia

import os
import getpass

os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter Your OpenAI API Key:")

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI

llm = OpenAI(temperature=0.1)

## Giving the Agent Tools
tools = load_tools(["ddg-search", "llm-math", "wikipedia"], llm=llm)

## Initialize the agent
# There are two ways you can instantiate the agent: AgentExecutor or initialize_agent.
# AgentExecutor - The AgentExecutor class is responsible for executing the agent's actions and managing the agent's memory.
#               - The AgentExecutor provides a more flexible and customizable way to run the agent, as you can specify the tools and memory to be used.
# When to use AgentExecutor - When you want more control over executing the agent’s actions and memory management.
#                           - When you want to specify the tools and memory to be used by the agent.

# initialize_agent - The initialize_agent function is a convenience function provided by LangChain that simplifies creating an agent.
#                  - It automatically initializes the agent with the specified language model and tools.

# When to use initialize_agent - When you want a simplified way to create an agent without specifying the memory.
#                              - When you want to create an agent with default settings, quickly.

agent = initialize_agent(tools,
                         llm,
                         agent="zero-shot-react-description",
                         verbose=True)

query = """
Who is the current Chief AI Scientist at Meta AI? When was he born?
What is his current age? What is the average life expectancy of people where he was born?
"""

agent.run(query)





