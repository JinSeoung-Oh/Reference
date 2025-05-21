### From https://generativeai.pub/building-ai-agents-with-reasoning-engine-on-vertex-ai-a-hands-on-guide-eb9e618f1154

# Import the required libraries
from vertexai.preview.generative_models import GenerativeModel
import vertexai
from langchain.agents import AgentExecutor, create_react_agent
from langchain_google_vertexai import ChatVertexAI
from langchain.tools import BaseTool, StructuredTool
from typing import Optional, Type
import math
import datetime
import json
import time

# Initialize Vertex AI
vertexai.init(project="your-project-id", location="your-region")

# Load the Gemini Pro model
model = GenerativeModel("gemini-pro")

# Define Python functions as tools
def calculate_square(number: float) -> float:
    """Calculate the square of a number.
    
    Args:
        number: The input number to square.
        
    Returns:
        The square of the input number.
    """
    return number ** 2

def calculate_square_root(number: float) -> float:
    """Calculate the square root of a number.
    
    Args:
        number: The input number to find the square root of.
        
    Returns:
        The square root of the input number.
    """
    if number < 0:
        return "Cannot calculate square root of a negative number"
    return math.sqrt(number)

# Convert functions to LangChain tools
tools = [
    StructuredTool.from_function(
        func=calculate_square,
        name="calculate_square",
        description="Calculate the square of a number",
    ),
    StructuredTool.from_function(
        func=calculate_square_root,
        name="calculate_square_root",
        description="Calculate the square root of a number",
    ),
]

# Create a Vertex AI chat model for LangChain
llm = ChatVertexAI(model_name="gemini-pro")

# Define the agent prompt
prompt = """You are a helpful AI assistant that can use tools to solve problems.
If you don't know how to answer a question, use the tools available to you.
Always show your reasoning step by step.
"""
# Create the ReAct agent
agent = create_react_agent(llm, tools, prompt)
# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Test the agent locally
result = agent_executor.invoke({"input": "What is the square root of 144, and then what is the square of that result?"})
print(result["output"])

-----------------------------------------------------------------------------
## Deploying the Agent to Vertex AI
# Define the agent app
from langchain_google_vertexai import VertexAIApp

# Create the Vertex AI app
app = VertexAIApp.from_agent(
    agent_executor=agent_executor,
    project="your-project-id",
    location="your-region",
    app_display_name="math-reasoning-agent",
)
# Deploy the app
app.deploy()

-----------------------------------------------------------------------------
## Model Configuration
# Customize model parameters
model = GenerativeModel(
    "gemini-pro",
    generation_config={
        "temperature": 0.2,
        "top_p": 0.8,
        "top_k": 40,
        "max_output_tokens": 1024,
    },
    safety_settings={
        "harassment": "block_medium_and_above",
        "hate_speech": "block_medium_and_above",
        "sexually_explicit": "block_medium_and_above",
        "dangerous_content": "block_medium_and_above",
    },
)

------------------------------------------------------------------------------
## Agent Configuration
# Customize agent configuration
prompt = """You are a specialized math assistant that can solve complex numerical problems.
Always approach problems step by step, showing your reasoning clearly.
Use the available tools when needed to perform calculations accurately.
If you're unsure about an approach, explain the different possibilities and why you chose a particular method.
"""

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5,
    early_stopping_method="generate",
)



