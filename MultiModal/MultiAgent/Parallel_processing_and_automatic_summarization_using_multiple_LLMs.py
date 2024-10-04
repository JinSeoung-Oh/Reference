## From https://medium.com/@astropomeai/multi-ai-agent-parallel-processing-and-automatic-summarization-using-multiple-llms-ad80f410ae21
## Just example

!pip install -qU langchain-google-genai
!pip install -qU langchain-anthropic
!pip install -qU langchain-core
!pip install -qU langchain-openai
!pip install -qU tavily-python
!pip install -qU langchain_community
!pip install -qU langgraph
!pip install -qU duckduckgo-search

import operator
from typing import Annotated, Any, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain.schema import HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
import os
from google.colab import userdata

os.environ["ANTHROPIC_API_KEY"] = userdata.get('ANTHROPIC_API_KEY')
os.environ["GOOGLE_API_KEY"] = userdata.get("GOOGLE_API_KEY")
os.environ['OPENAI_API_KEY'] = userdata.get('OPENAI_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = userdata.get('LANGCHAIN_TRACING_V2')
os.environ['LANGCHAIN_API_KEY'] = userdata.get('LANGCHAIN_API_KEY')

class State(TypedDict):
    messages: Annotated[List[str], operator.add]
    iteration: int

class Node:
    def __init__(self, name: str):
        self.name = name

    def process(self, state: State) -> dict:
        raise NotImplementedError

    def __call__(self, state: State) -> dict:
        result = self.process(state)
        print(f"{self.name}: Processing complete")
        return result

class SupervisorNode(Node):
    def __init__(self):
        super().__init__("Supervisor")

    def process(self, state: State) -> dict:
        state['iteration'] = state.get('iteration', 0) + 1
        task = state['messages'][0] if state['messages'] else "No task set."
        return {"messages": [f"Iteration {state['iteration']}: {task}"]}

class AgentNode(Node):
    def __init__(self, agent_name: str, llm):
        super().__init__(agent_name)
        self.llm = llm

    def process(self, state: State) -> dict:
        task = state['messages'][-1]
        response = self.llm.invoke([HumanMessage(content=f"You are {self.name}. Please execute the following task: {task}")])
        return {"messages": [f"{self.name}'s response: {response.content}"]}

class SummarizerNode(Node):
    def __init__(self, llm):
        super().__init__("Summarizer")
        self.llm = llm

    def process(self, state: State) -> dict:
        responses = state['messages'][1:]  # Skip the initial task
        summary_request = "Please summarize the responses from each agent so far.\n" + "\n".join(responses)
        response = self.llm.invoke([HumanMessage(content=summary_request)])
        return {"messages": [f"Summary: {response.content}"]}

def create_llm(model_class, model_name, temperature=0.7):
    return model_class(model_name=model_name, temperature=temperature)

# High-performance models
claude = create_llm(ChatAnthropic, "claude-3-5-sonnet-20240620")
gemini = create_llm(ChatGoogleGenerativeAI, "gemini-1.5-pro-002")
openai = create_llm(ChatOpenAI, "gpt-4o")

# Lightweight models
# claude = create_llm(ChatAnthropic, "claude-3-haiku-20240307")
# gemini = create_llm(ChatGoogleGenerativeAI, "gemini-1.5-flash-latest")
# openai = create_llm(ChatOpenAI, "gpt-4o-mini")

builder = StateGraph(State)

nodes = {
    "supervisor": SupervisorNode(),
    "agent1": AgentNode("Agent1 (claude)", claude),
    "agent2": AgentNode("Agent2 (gemini)", gemini),
    "agent3": AgentNode("Agent3 (gpt)", openai),
    "summarizer": SummarizerNode(openai)
}

for name, node in nodes.items():
    builder.add_node(name, node)

builder.add_edge(START, "supervisor")
for agent in ["agent1", "agent2", "agent3"]:
    builder.add_edge("supervisor", agent)
    builder.add_edge(agent, "summarizer")
builder.add_edge("summarizer", END)

graph = builder.compile()

## Checking the Graph Definition
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))

