### From https://ai.gopubby.com/langgraph-building-a-dynamic-order-management-system-a-step-by-step-tutorial-0be56854fc91

import os
import pandas as pd
import random
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.runnables.graph import MermaidDrawMethod
from IPython.display import display, Image
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from typing import Dict, TypedDict

# Load environment variables
os.environ["OPENAI_API_KEY"] = ""

# Load datasets
inventory_df = pd.read_csv("inventory.csv")
customers_df = pd.read_csv("customers.csv")

# Convert datasets to dictionaries
inventory = inventory_df.set_index("item_id").T.to_dict()
customers = customers_df.set_index("customer_id").T.to_dict()

class State(TypedDict):
    query: str
    category: str
    next_node: str
    item_id: str
    order_status: str
    cost: str
    payment_status: str
    location: str
    quantity: int

def cancel_order(query: str) -> dict:
    """Simulate order cancelling"""
    order_id = llm.with_structured_output(method='json_mode').invoke(f'Extract order_id from the following text in json format: {query}')['order_id']
    #amount = query.get("amount")

    if not order_id:
        return {"error": "Missing 'order_id'."}

    return {"order_status": "Order stands cancelled"}


# Initialize LLM and bind tools
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

tools_2 = [cancel_order]
llm_with_tools_2 = llm.bind_tools(tools_2)
tool_node_2 = ToolNode(tools_2)

def call_model_2(state: MessagesState):
    """Use the LLM to decide the next step."""
    messages = state["messages"]
    response = llm_with_tools_2.invoke(str(messages))
    return {"messages": [response]}

def call_tools_2(state: MessagesState) -> Literal["tools_2", END]:
    """Route workflow based on tool calls."""
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.tool_calls:
        return "tools_2"
    return END

def categorize_query(state: MessagesState) -> MessagesState:
    """Categorize user query into PlaceOrder or CancelOrder"""
    prompt = ChatPromptTemplate.from_template(
        "Categorize user query into PlaceOrder or CancelOrder"
        "Respond with either 'PlaceOrder', 'CancelOrder' Query: {state}"
    )

    chain = prompt | ChatOpenAI(temperature=0)
    category = chain.invoke({"state": state}).content
    
    return {"query":state,"category": category}

def check_inventory(state: MessagesState) -> MessagesState:
    """Check if the requested item is in stock."""

    item_id = llm.with_structured_output(method='json_mode').invoke(f'Extract item_id from the following text in json format: {state}')['item_id']
    quantity = llm.with_structured_output(method='json_mode').invoke(f'Extract quantity from the following text in json format: {state}')['quantity']

    if not item_id or not quantity:
        return {"error": "Missing 'item_id' or 'quantity'."}

    if inventory.get(item_id, {}).get("stock", 0) >= quantity:
        print("IN STOCK")
        return {"status": "In Stock"}
    return {"query":state,"order_status": "Out of Stock"}

def compute_shipping(state: MessagesState) -> MessagesState:
    """Calculate shipping costs."""
    item_id = llm.with_structured_output(method='json_mode').invoke(f'Extract item_id from the following text in json format: {state}')['item_id']
    quantity = llm.with_structured_output(method='json_mode').invoke(f'Extract quantity from the following text in json format: {state}')['quantity']
    customer_id = llm.with_structured_output(method='json_mode').invoke(f'Extract customer_id from the following text in json format: {state}')['customer_id']
    location = customers[customer_id]['location']


    if not item_id or not quantity or not location:
        return {"error": "Missing 'item_id', 'quantity', or 'location'."}

    weight_per_item = inventory[item_id]["weight"]
    total_weight = weight_per_item * quantity
    rates = {"local": 5, "domestic": 10, "international": 20}
    cost = total_weight * rates.get(location, 10)
    print(cost,location)

    return {"query":state,"cost": f"${cost:.2f}"}

def process_payment(state: State) -> State:
    """Simulate payment processing."""
    cost = llm.with_structured_output(method='json_mode').invoke(f'Extract cost from the following text in json format: {state}')

    if not cost:
        return {"error": "Missing 'amount'."}
    print(f"PAYMENT PROCESSED: {cost} and order successfully placed!")
    payment_outcome = random.choice(["Success", "Failed"])
    return {"payment_status": payment_outcome}

def route_query_1(state: State) -> str:
    """Route the query based on its category."""
    print(state)
    if state["category"] == "PlaceOrder":
        return "PlaceOrder"
    elif state["category"] == "CancelOrder":
        return "CancelOrder"

# Create the workflow
workflow = StateGraph(MessagesState)

#Add nodes
workflow.add_node("RouteQuery", categorize_query)
workflow.add_node("CheckInventory", check_inventory)
workflow.add_node("ComputeShipping", compute_shipping)
workflow.add_node("ProcessPayment", process_payment)

workflow.add_conditional_edges(
    "RouteQuery",
    route_query_1,
    {
        "PlaceOrder": "CheckInventory",
        "CancelOrder": "CancelOrder"
    }
)
workflow.add_node("CancelOrder", call_model_2)
workflow.add_node("tools_2", tool_node_2)


# Define edges

workflow.add_edge(START, "RouteQuery")
workflow.add_edge("CheckInventory", "ComputeShipping")
workflow.add_edge("ComputeShipping", "ProcessPayment")
workflow.add_conditional_edges("CancelOrder", call_tools_2)
workflow.add_edge("tools_2", "CancelOrder")
workflow.add_edge("ProcessPayment", END)

# Compile the workflow
agent = workflow.compile()

# Visualize the workflow
mermaid_graph = agent.get_graph()
mermaid_png = mermaid_graph.draw_mermaid_png(draw_method=MermaidDrawMethod.API)
display(Image(mermaid_png))

# Query the workflow
user_query = "I wish to cancel order_id 223"
for chunk in agent.stream(
    {"messages": [("user", user_query)]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print() 

auser_query = "customer_id: customer_14 : I wish to place order for item_51 with order quantity as 4 and domestic"
for chunk in agent.stream(
    {"messages": [("user", auser_query)]},
    stream_mode="values",
):
    chunk["messages"][-1].pretty_print()

