# From https://medium.com/@Shrishml/a-primer-on-ai-agents-with-langgraph-understand-all-about-it-0534345190dc
"""
Building AI Agents

There are four fundamental design patterns that form the building blocks of most agents. 
These patterns can be used independently, combined, or interwoven depending on the specific use case.

1. Planning: 
   Effective planning is critical for all agents. It involves breaking down complex problems into smaller, 
   more manageable steps that can be reasoned about using available tools. 
   This requires the agent to have a working understanding of its environment and the tools at its disposal.
2. Reflection:
   Agents learn and adapt by reflecting on the results and feedback received from their interactions with the environment. 
   By analyzing these outcomes, the agent can adjust its approach or problem-solving strategies for future encounters.
3. Tool Use:
   Unlike humans who can manipulate the physical world directly, LLMs (Large Language Models) 
   often rely on external tools to interact with their environment. 
   These tools allow them to perform actions beyond simple communication, much like how humans use tools to accomplish tasks in the real world.
4. Multi-agent Collaboration:
   This emerging pattern involves dividing tasks among different types of agents, each specialized for a specific function. 
   This collaborative approach mirrors human organizations where teams of experts in areas 
   like HR, finance, and technology work together to achieve a common goal.

An AI agent is built upon four key components:

1. Brain (Decision-Making): 
   This component, often powered by Large Language Models (LLMs), acts as the agent’s “brain.” 
   It analyzes the environment, interprets information, and formulates plans to achieve the agent’s goals.
2. Memory: 
   The agent’s memory stores crucial information gathered during operation. 
   This data can include past experiences, environmental details, and learned patterns, all of which are used to inform future decisions.
3. Workflow (Action Management):
   This component dictates the order and flow of the agent’s actions. 
   Some constraints or rules are introduced within the workflow to ensure the agent operates reliably and achieves its goals efficiently.
4. Tools: 
   These are the external capabilities the agent can leverage to interact with the environment beyond simple communication. 
   Tools allow the agent to perform actions and complete tasks in the real world.
As we move forward, we’ll explore how each of these components can be implemented using Langgraph.

For now, we are good at managing memory and tool execution (openai function calling), but workflow remains a challenge. 
There are two main approaches:

1. Free-flowing LLM control: 
   This lets the agent determine its own control flow, similar to function-calling and ReACT agents in Langchain.
   However, this approach has limitations as it deviates and takes relatively more time.
2. Predefined control flow:
   This involves defining the workflow beforehand using structures like DAGs (Directed Acyclic Graphs) or cycles. 
   While Langchain LCEL can handle DAGs with conditionals as well, it cannot implement cycles. 
   In Langgraph, we can predefined complete workflow with conditional and cyclic components well, providing much flexibility to agents.

LangGraph simplifies AI agent development by focusing on three key components:

1. State: 
   This represents the agent’s current information, often stored as a dictionary or managed through a database.
2. Nodes: 
   These are the building blocks that execute computations. They can be LLM-based, Python code, or even other Langgraph units (subgraphs). 
   Nodes take the state as input, perform modifications, and return the updated state.
3. Edges:
   Edges define the agent’s control flow. They dictate the specific path the agent follows, 
   but Langgraph injects flexibility through conditional nodes and cycles. 
   These nodes allow the agent to adapt its course based on specific conditions met within the graph in the shared State.

The Planner Node: Orchestration Efficiency

While the planner node appears complex, its core functionalities are designed for efficiency.
It acts as the maestro of the orchestration process, taking on three key roles:

1. Initiation Planning: 
   At the beginning of the workflow, the planner crafts an initial plan outlining the steps the agent needs to take to address the user’s query. 
   This plan is stored in the state variable, which is a list.
2. Adaptive Planning:
   As the workflow progresses, the planner can dynamically update the plan based on two factors: 
   intermediate results and past plans. This adaptability allows the agent to react to unforeseen circumstances and refine its approach as needed.
   When the planner returns the steps list, it efficiently adds new steps to the existing list instead of overwriting everything.
3. Completion Signal: Finally, the planner plays a crucial role in declaring the end of the plan. 
   It analyzes intermediate results and past plans to determine if all necessary steps have been completed. 
   If so, the planner simply updates the end field in the state to True, signaling the workflow's completion.
"""

## Example
import os
from dotenv import load_dotenv
from pathlib import Path
dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path

import pandas as pd

from langchain.tools import tool

@tool
def addition(x, y):
    """Addition of two number
    :param: x: The first number to be added 
    :param: y: The second number to be added"""

    return x+y

@tool
def subtraction(x, y):
    """Sumbtration of two number
    :param: x: The first number the greater one  
    :param: y: The second number to be subtracted """

    return x-y

@tool
def multiplication(x, y):
    """Multiplication of two number
    :param: x: The first number to be multiplied 
    :param: y: The second number to be multiplied"""

    return x*y

@tool
def division(x, y):
    """Division of two number
    :param: x: The first number the greater one  
    :param: y: The second number to be devided """

    return x/y

tools = [addition, subtraction, multiplication, division]

tool_dict =  {} # this is going to be required during tool execution

for tool in tools:
    tool_dict[tool.name]= tool

import json
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
from typing import List, Optional
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers.openai_tools import JsonOutputToolsParser
import operator
import os 
from loguru import logger

class StrategyAgentState(TypedDict):
    user_query: str
    steps: Annotated[List, operator.add]
    step_no: int
    results: dict
    final_response: str
    end:bool

deployment_name = os.environ["AZURE_DEPLOYMENT"]
azure_endpoint = os.environ["AZURE_ENDPOINT"]
api_key = os.environ["AZURE_API_KEY"]
api_version = os.environ["API_VERSION"]

llm = AzureChatOpenAI(azure_deployment=deployment_name,
                            azure_endpoint=azure_endpoint,
                            api_key=api_key,
                            api_version=api_version, temperature=0.0)

def plan(state: StrategyAgentState):
    """The planner node, this is the brain of the system"""
    user_question = state["user_query"]
    steps = state["steps"]
    results = state["results"]
    end = state["end"]

    if results is None:  # If result has not been populated yet we will start planning 
        SYSTEM_PROMT = "You are a helpful assitant who is good is mathematics.\
            Do not calculate yourself let the tool do the calculation. Call one tool at a time"
        prompt_template = ChatPromptTemplate.from_messages(
                        [("system", SYSTEM_PROMT),
                        ("user", "{user_question}")])

        planner = prompt_template | llm.bind_tools(tools)| JsonOutputToolsParser()

    

        invoke_inputs = {"user_question": user_question}
        steps = planner.invoke(invoke_inputs)

        logger.info(f"Generated plans : {steps}")

        return {'steps': steps}
    elif results and not end: # If result has been populated and end is not true we will go to end detector
        SYSTEM_PROMT = "You need to decide whether a problem is solved or not. Just return ##YES if propblem is solved and ##NO \
        if problem is not solved. Please expalain your reasoning as well. Make sure you use same template of ##YES and ##NO in final answer.\
         Do not calculate yourself let the tool do the calculation"
        prompt_template = ChatPromptTemplate.from_messages(
                        [("system", SYSTEM_PROMT),
                        ("user", "{user_question}"), 
                        ("user", "{results}"),
                        ("user", "{steps}")])

        planner = prompt_template | llm

    

        invoke_inputs = {"user_question": user_question, "steps":json.dumps(steps), "results":json.dumps(results)}
        response = planner.invoke(invoke_inputs)

        logger.info(f"End detector response : {response.content}")

        if  "##YES" in response.content:
            return {'end': True}
        elif "##NO" in response.content:
            return {'end': False}
    else: # if end is not true and 
        SYSTEM_PROMT = "You are a helpful assitant who is good is mathematics.\
              You are replanner assistant.\
        If you are given previous steps and previous results. Do not start again. Call one function at a time.\
             Do not calculate yourself let the tool do the calculation"
        prompt_template = ChatPromptTemplate.from_messages(
                        [("system", SYSTEM_PROMT),
                        ("user", "{user_question}"), 
                        ("user", "{steps}"),
                        ("user", "{results}")])

        planner = prompt_template | llm.bind_tools(tools)| JsonOutputToolsParser()

    

        invoke_inputs = {"user_question": user_question, "steps":json.dumps(steps), "results":json.dumps(results)}
        steps = planner.invoke(invoke_inputs)

        logger.info(f"Pending  plans : {steps}")

        return {'steps': steps}

def tool_execution(state: StrategyAgentState):

    """ Worker node that executes the tools of a given plan. Plan is json arguments
    which can be sent to tools directly"""

    steps = state["steps"]
    step_no = state["step_no"] or 0


    _results = state["results"] or {}
    j= 0
    for tool in steps[step_no: ]:

        tool_name = tool['type']
        args = tool["args"]
        _results[tool_name+"_step_"+str(step_no+j)] = tool_dict[tool_name](args)
        logger.info(f"{tool_name} is called with arguments {args}")
        j=j+1

    return {"results": _results, "step_no": step_no+j, }

def responder(state:StrategyAgentState):


    user_question = state["user_query"]
    results = state["results"]
    SYSTEM_PROMT = "Generate final response by looking at the results and original user question."
    prompt_template = ChatPromptTemplate.from_messages(
                    [("system", SYSTEM_PROMT),
                     ("user", "{user_question}"),
                     ("user", "{results}")])

    model = prompt_template | llm

 

    invoke_inputs = {"user_question": user_question, "results": json.dumps(results)}
    response = model.invoke(invoke_inputs)
    return {"final_response": response.content}

def route(state:StrategyAgentState):
    """A conditional route based on number of steps completed or end anounced by any other node,
      this will either end the execution or will be sent to tools for planning"""

    steps = state["steps"]
    step_no = state["step_no"]
    end = state["end"]
    if end:
        # We have executed all tasks
        return "respond"
    else:
        # We are still executing tasks, loop back to the "tool" node
        return "plan"

graph = StateGraph(StrategyAgentState)
graph.add_node("plan", plan)
graph.add_node("tool_execution", tool_execution)
graph.add_node("responder", responder)
#--------------------------------------------------------
graph.add_edge("plan", "tool_execution")
graph.add_edge("responder", END)
graph.add_conditional_edges("tool_execution", route, {"respond":"responder", "plan":"plan"})
graph.set_entry_point("plan")
agent = graph.compile()

query = "what is 3 multiplied by 9 added to 45 then devide all by 6"


for s in agent.stream({"user_query": query}):
    print(s)
    print("--------------------")
