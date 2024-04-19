"""
# From https://medium.com/@ssmaameri/building-a-simple-agent-with-tools-and-toolkits-in-langchain-77e0f9bd1fa5

"""
mkdir simple-math-agent && cd simple-math-agent
touch math-agent.py
python3 -m venv .venv
. .venv/bin/activate

pip install langchain langchain_openai

git clone git@github.com:smaameri/simple-math-agent.git

## The Tool
# Basic way to define tool with lanchain 
from langchain_core.tools import BaseTool

class AddTool(BaseTool):
    name = "add"
    description = "Adds two numbers together"
    args_schema: Type[BaseModel] = AddInput
    return_direct: bool = True

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        return a + b

# Easy way to define tool wih lanchain
from langchain.tools import tool

@tool
def add(a: int, b: int) -> int:
 “””Adds two numbers together””” # this docstring gets used as the description
 return a + b # the actions our tool performs

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b

@tool
def square(a) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

## The toolkit
toolkit = [add, multiply, square]

## OpenAI with custom agnet toolkit
import os

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI

from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

os.environ["OPENAI_API_KEY"] = "sk-"

# setup the tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


@tool
def square(a) -> int:
    """Calculates the square of a number."""
    a = int(a)
    return a * a

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a mathematical assistant.
        Use your tools to answer questions. If you do not have a tool to
        answer the question, say so. 

        Return only the answers. e.g
        Human: What is 1 + 1?
        AI: 2
        """),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Choose the LLM that will drive the agent
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)

# setup the toolkit
toolkit = [add, multiply, square]

# Construct the OpenAI Tools agent
agent = create_openai_tools_agent(llm, toolkit, prompt)

# Create an agent executor by passing in the agent and tools
agent_executor = AgentExecutor(agent=agent, tools=toolkit, verbose=True)

result = agent_executor.invoke({"input": "what is 1 + 1?"})

print(result['output'])




