### From https://ai.plainenglish.io/building-smart-ai-agents-b48ab6e83bc7
"""
Summary: Implementing the ReAct Pattern with Traditional Methods vs. LangGraph

The ReAct (Reasoning and Acting) pattern enhances AI agents by interleaving reasoning with action-taking, 
following a loop of Thought → Action → Observation until a final answer is reached. 
This summary compares a traditional Python implementation of the ReAct pattern with a modern, structured approach using LangGraph, 
illustrated through a restaurant rating comparison example.

1. Understanding the ReAct Pattern
   -a. Loop Structure:
       -1. Thought: Agent reasons about the next steps.
       -2. Action: Agent performs an action (e.g., call a function).
       -3. Observation: Agent observes the result of its action.
       -4. Repeat until final answer is determined.

2. Traditional Implementation Highlights
   -a. State Management: Maintains conversation history in a list of messages.
   -b. System Prompt: Defines agent behavior and available actions (e.g., fetching restaurant ratings).
   -c. Action Handling:
       -1. Action Functions: Functions like get_restaurant_rating perform tasks.
       -2. Action Registry: A dictionary (known_actions) maps action names to functions.
   -d. Main Query Loop:
       -1. Uses regular expressions to detect actions from the agent's output.
       -2. Executes matched actions, fetches observations, and feeds them back to the agent.
       -3. Continues the loop until no more actions are needed.
   -e. Pros:
       -1. Simple to understand and implement.
       -2. Good for prototyping and straightforward tasks.
   -f. Cons:
       -1. Code can become complex as workflows grow.
       -2. State and flow management less structured, making debugging and scaling harder.

3. LangGraph Implementation Highlights
   LangGraph offers a graph-based, structured approach to implementing the ReAct pattern, improving modularity, state management, and maintainability.
   
   -a. Typed State Management: Uses TypedDict for clear, type-safe state definitions.
   -b. Graph-Based Flow:
       -1. Nodes: Pure functions that process and update state.
       -2. Edges & Conditional Routing: Explicitly define flow between nodes, including decision points.
       -3. Compilation: Optimizes the graph for execution, ensuring efficiency.
    -c. Tool Integration: Tools like RestaurantTool are wrapped as LangChain tools and integrated into the agent.
    -d. State Flow:
        -1. LLM Node: Calls the language model with current messages.
        -2. Conditional Edge: Decides if an action should be taken based on LLM output.
        -3. Action Node: Executes tool actions, updates state with observations.
        -4. Loop Back: Returns control to the LLM node for further reasoning if needed.
    -e. Pros:
        -1. Structured Flow: Explicit graph structure makes logic clear.
        -2. Robust State Management: Typed states catch errors early and document expected data.
        -3. Modularity and Extensibility: Adding or modifying capabilities is easier with well-defined nodes.
        -4. Visualization and Debugging: Graph structure aids in understanding agent decision processes.
        -5. Scalable for Production: Better suited for complex workflows and team-based development.
    -f. Cons:
        -1. Slightly steeper learning curve compared to simple, traditional scripts.
        -2. Might be overkill for simple, one-off tasks or quick prototypes.

4. Key Differences and Benefits
   -a. Traditional Approach: Ideal for quick prototyping, simple applications, and learning purposes. 
                             It focuses on straightforward loops and regex parsing for actions but can become unwieldy as complexity grows.
   -b. LangGraph Approach: Offers a robust, scalable, and maintainable framework for production applications. 
                           Its graph-based architecture provides explicit control flow, clear state management, modularity, and enhanced debugging capabilities.

5. Use Cases
   -a. Traditional ReAct: Best for small-scale projects, rapid testing, and educational examples.
   -b. LangGraph ReAct: Suited for complex workflows like customer service automation, data analysis pipelines, decision-making systems,
       and any production environment requiring maintainability and scalability.

6. Conclusion:
   Both traditional and LangGraph implementations demonstrate the effectiveness of the ReAct pattern in AI agents. 
   The traditional approach is excellent for simple, quick implementations, while LangGraph brings structure, robustness, 
   and scalability for more complex, production-level systems. The choice depends on the project's complexity, scalability needs, 
   and development context.
"""
##### Traditional Implementation
import re
from openai import OpenAI
from dotenv import load_dotenv

_ = load_dotenv()

class RestaurantAgent:
    def __init__(self, system=""):
        self.system = system
        self.messages = []
        self.client = OpenAI()
        if self.system:
            self.messages.append({'role': 'system', 'content': system})

    def __call__(self, message):
        self.messages.append({'role': 'user', 'content': message})
        result = self.execute()
        self.messages.append({'role': 'assistant', 'content': result})
        return result
    
    def execute(self):
        completion = self.client.chat.completions.create(
            model='gpt-4',
            temperature=0,
            messages=self.messages
        )
        return completion.choices[0].message.content

def get_restaurant_rating(name):
    ratings = {
        "Pizza Palace": {"rating": 4.5, "reviews": 230},
        "Burger Barn": {"rating": 4.2, "reviews": 185},
        "Sushi Supreme": {"rating": 4.8, "reviews": 320}
    }
    return ratings.get(name, {"rating": 0, "reviews": 0})

known_actions = {
    "get_rating": get_restaurant_rating
}

prompt = """
You run in a loop of Thought, Action, PAUSE, Observation.
At the end of the loop you output an Answer.
Use Thought to describe your reasoning about the restaurant comparison.
Use Action to run one of the actions available to you - then return PAUSE.
Observation will be the result of running those actions.

Your available actions are:
get_rating:
e.g. get_rating: Pizza Palace
Returns rating and review count for the specified restaurant

Example session:
Question: Which restaurant has better ratings, Pizza Palace or Burger Barn?
Thought: I should check the ratings for both restaurants
Action: get_rating: Pizza Palace
PAUSE
"""

def query(question, max_turns=5):
    action_re = re.compile('^Action: (\w+): (.*)$')
    bot = RestaurantAgent(prompt)
    next_prompt = question
    
    for i in range(max_turns):
        result = bot(next_prompt)
        print(result)
        actions = [
            action_re.match(a) 
            for a in result.split('\n') 
            if action_re.match(a)
        ]
        if actions:
            action, action_input = actions[0].groups()
            if action not in known_actions:
                raise Exception(f"Unknown action: {action}: {action_input}")
            observation = known_actions[action](action_input)
            next_prompt = f"Observation: {observation}"
        else:
            return
        
question = """which resturant have better rating, Pizza Palace or Burger Barn?"""
query(question)

####### Modern Implementation with LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
from langchain_core.messages import (
    AnyMessage, 
    SystemMessage, 
    HumanMessage, 
    ToolMessage,
    AIMessage
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# Load environment variables properly
load_dotenv()

# Define a more structured prompt template with tool descriptions
prompt_template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!
Question: {input}
Thought: {agent_scratchpad}"""

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]

class RestaurantTool:
    def __init__(self):
        self.name = "restaurant_rating"
        self.description = "Get rating and review information for a restaurant"
    
    def get_restaurant_rating(self, name: str) -> dict:
        ratings = {
            "Pizza Palace": {"rating": 4.5, "reviews": 230},
            "Burger Barn": {"rating": 4.2, "reviews": 185},
            "Sushi Supreme": {"rating": 4.8, "reviews": 320}
        }
        return ratings.get(name, {"rating": 0, "reviews": 0})

    def __call__(self, name: str) -> str:
        result = self.get_restaurant_rating(name)
        return f"Rating: {result['rating']}/5.0 from {result['reviews']} reviews"

class Agent:
    def __init__(self, model: ChatOpenAI, tools: List[Tool], system: str = ''):
        self.system = system
        self.tools = {t.name: t for t in tools}
        
        # Create tool descriptions for the prompt
        tool_descriptions = "\n".join(f"- {t.name}: {t.description}" for t in tools)
        tool_names = ", ".join(t.name for t in tools)
        
        # Bind tools to the model
        self.model = model.bind_tools(tools)
        
        # Initialize the graph
        graph = StateGraph(AgentState)
        
        # Add nodes and edges
        graph.add_node("llm", self.call_llm)
        graph.add_node("action", self.take_action)
        
        # Add conditional edges
        graph.add_conditional_edges(
            "llm",
            self.should_continue,
            {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        
        # Set entry point and compile
        graph.set_entry_point("llm")
        self.graph = graph.compile()

    def should_continue(self, state: AgentState) -> bool:
        """Check if there are any tool calls to process"""
        last_message = state["messages"][-1]
        return hasattr(last_message, "tool_calls") and bool(last_message.tool_calls)

    def call_llm(self, state: AgentState) -> AgentState:
        """Process messages through the LLM"""
        messages = state["messages"]
        if self.system and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system)] + messages
        response = self.model.invoke(messages)
        return {"messages": [response]}

    def take_action(self, state: AgentState) -> AgentState:
        """Execute tool calls and return results"""
        last_message = state["messages"][-1]
        results = []
        
        for tool_call in last_message.tool_calls:
            tool_name = tool_call['name']
            if tool_name not in self.tools:
                result = f"Error: Unknown tool '{tool_name}'"
            else:
                try:
                    tool_result = self.tools[tool_name].invoke(tool_call['args'])
                    result = str(tool_result)
                except Exception as e:
                    result = f"Error executing {tool_name}: {str(e)}"
            
            results.append(
                ToolMessage(
                    tool_call_id=tool_call['id'],
                    name=tool_name,
                    content=result
                )
            )
        
        return {"messages": results}

    def invoke(self, message: str) -> List[AnyMessage]:
        """Main entry point for the agent"""
        initial_state = {"messages": [HumanMessage(content=message)]}
        final_state = self.graph.invoke(initial_state)
        return final_state["messages"]

# Create and configure the agent
def create_restaurant_agent() -> Agent:
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # Create tool instance
    restaurant_tool = RestaurantTool()
    
    # Convert to LangChain Tool
    tool = Tool(
        name=restaurant_tool.name,
        description=restaurant_tool.description,
        func=restaurant_tool
    )
    
    # Create system prompt
    system_prompt = prompt_template.format(
        tools=tool.description,
        tool_names=tool.name,
        input="{input}",
        agent_scratchpad="{agent_scratchpad}"
    )
    
    # Create and return agent
    return Agent(model, [tool], system=system_prompt)

# Example usage
if __name__ == "__main__":
    agent = create_restaurant_agent()
    response = agent.invoke("""which resturant have better rating, Pizza Palace or Burger Barn?""")
    for message in response:
        print(f"{message.type}: {message.content}"))



