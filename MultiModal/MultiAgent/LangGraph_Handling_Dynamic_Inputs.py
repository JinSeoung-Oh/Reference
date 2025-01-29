### From https://harshaselvi.medium.com/building-ai-agents-using-langgraph-part-7-handling-dynamic-inputs-and-human-interrupts-af33869cd3cb

from typing_extensions import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AnyMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv

load_dotenv()

# Define the state for the graph
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# Tool: Person Information
def get_person_details(person_name: str) -> str:
    """Retrieve details of a person."""
    return f"{person_name} is a DevOps Engineer."

# Tool: Location Information
def get_person_location(person_name: str) -> str:
    """Retrieve location of a person."""
    return f"{person_name} lives in Bangalore."

# Initialize ChatGroq model
llm = ChatGroq(model="llama-3.1-8b-instant")
# Bind tools to the Groq model
tools = [get_person_details, get_person_location]
llm_with_tools = llm.bind_tools(tools)

# Define the assistant node
def assistant(state: MessagesState):
    sys_msg = SystemMessage(
        "You are a helpful assistant who answers questions accurately using the available tools.")
    return {"messages": llm_with_tools.invoke([sys_msg] + state["messages"])}

def ask_human_input(prompt: str) -> str:
    """Helper function to prompt user input."""
    return input(f"{prompt}\n> ")

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools=tools))
# Add graph edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
# Compile the graph with human interrupts
graph = builder.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["assistant"],
    interrupt_after=[],
)

# Simulate interaction
thread = {"configurable": {"thread_id": "1"}}
# Initialize state with a message
graph.invoke({"messages": [HumanMessage("Who is Harsha?")]}, thread)
# Ask for user input and update the state
user_question = ask_human_input(
    "Your question: Who is Harsha?\nDo you want to ask anything else about Harsha?")
user_message = HumanMessage(content=user_question)
# Update state with the new user message
graph.update_state(thread, {"messages": [user_message]})
# Stream the result
result = graph.stream(None, thread, stream_mode="values")
# Display the result
for event in result:
    event["messages"][-1].pretty_print()


