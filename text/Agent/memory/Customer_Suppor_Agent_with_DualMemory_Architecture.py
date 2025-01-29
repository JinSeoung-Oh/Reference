### From https://ai.gopubby.com/building-a-customer-support-agent-with-dual-memory-architecture-long-and-short-term-memory-c39ab176046e

import os

from dotenv import load_dotenv
import os
import uuid
from langgraph.store.memory import InMemoryStore
from langchain_anthropic import ChatAnthropic
from IPython.display import Image, display

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.store.base import BaseStore

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig

os.environ["LANGCHAIN_PROJECT"] = "langgraph_store_customer_support_agent"
os.environ["LANGCHAIN_TRACING_V2"] = "true"

in_memory_store = InMemoryStore()

# Namespace for the memory to save
customer_id = "1"
namespace_for_memory = (customer_id, "customer_interactions")

# Save a memory to namespace as key and value pairs
key = str(uuid.uuid4())

# The values needs to be a dictionary  
value = {"name" : "John Doe", "email": "johndoe@example.com"}

# Save the memory on disk
in_memory_store.put(namespace_for_memory, key, value)

# Delete memory from disk
in_memory_store.delete(namespace=namespace_for_memory, key=key)

# Search 
memories = in_memory_store.search(namespace_for_memory)
type(memories)

# Memory metatdata 
memories[0].dict()

# create model
model = ChatAnthropic(
    temperature=0.0,
    model="claude-3-opus-20240229",
    api_key=os.environ.get("ANTHROPIC_API_KEY")
)

COMPANY_NAME = "Code With Prince PLC"

MODEL_SYSTEM_MESSAGE = """You are a helpful customer support assistant for {company_name}. 
Use the customer's history to provide relevant and personalized support.
Customer profile: {history}"""

CREATE_HISTORY_INSTRUCTION = """Update the customer profile with new support interaction details.

CURRENT PROFILE:
{history}

ANALYZE FOR:
1. Contact history
2. Product usage/purchases
3. Previous issues/resolutions 
4. Preferences (communication, products)
5. Special circumstances

Focus on verified support interactions only. Summarize key details clearly.

Update profile based on this conversation:"""

def call_model(state: MessagesState, config: RunnableConfig, store: BaseStore):
   """Generates AI response using customer context and history.
   
   Args:
       state: Current conversation messages
       config: Runtime configuration with customer_id
       store: Persistent storage for customer data
       
   Returns:
       dict: Generated response messages
       
   Flow:
       1. Gets customer profile from store using ID
       2. Formats system prompt with customer context
       3. Generates personalized response
   """
   # Get customer ID and profile from store
   customer_id = config["configurable"]["customer_id"]
   namespace = ("customer_interactions", customer_id)
   key = "customer_data_memory"
   memory = store.get(namespace, key)

   # Extract interaction history or set default for new customers
   history = memory.value.get('customer_data_memory') if memory else "No existing memory found."

   # Generate response with customer context
   system_msg = MODEL_SYSTEM_MESSAGE.format(history=history, company_name=COMPANY_NAME)
   response = model.invoke([SystemMessage(content=system_msg)] + state["messages"])

   return {"messages": response}

def write_memory(state: MessagesState, config: RunnableConfig, store: BaseStore):
   """Updates customer interaction history in persistent storage.
   
   Args:
       state: Current conversation messages
       config: Runtime config containing customer_id 
       store: Persistent storage for customer data
       
   Flow:
       1. Retrieves existing customer history
       2. Analyzes conversation for new insights
       3. Updates stored history with new data
   """
   # Get customer history
   user_id = config["configurable"]["customer_id"]
   namespace = ("customer_interactions", user_id)
   key = "customer_data_memory"
   memory = store.get(namespace=namespace, key=key)
   
   # Extract existing history or set default
   history = memory.value.get(key) if memory else "No existing history."

   # Generate and store updated history
   system_msg = CREATE_HISTORY_INSTRUCTION.format(history=history)
   new_insights = model.invoke([SystemMessage(content=system_msg)] + state['messages'])
   store.put(namespace, key, {"customer_interactions": new_insights.content})

# Build conversational AI system with memory persistence
builder = StateGraph(MessagesState)

# Add core processing nodes
builder.add_node("call_model", call_model)     # AI response generator node
builder.add_node("write_memory", write_memory) # Memory persistence node

# Configure processing flow
builder.add_edge(START, "call_model")          # Initial: Generate response
builder.add_edge("call_model", "write_memory") # Next: Save conversation context
builder.add_edge("write_memory", END)          # Finally: Complete interaction cycle

# Initialize memory stores
across_thread_memory = InMemoryStore()         # Long-term customer history storage
within_thread_memory = MemorySaver()           # Current conversation buffer

# Compile graph with memory configuration
graph = builder.compile(
   checkpointer=within_thread_memory,         # Track conversation state
   store=across_thread_memory                 # Persist customer interactions
)

within_thread_memory = MemorySaver()  # Current conversation buffer

# Initialize conversation configuration
config = {
   "configurable": {
       "thread_id": "1",    # Current conversation ID
       "customer_id": "1"   # Customer profile ID
   }
}

# Set initial customer message
input_msg = [HumanMessage(content="Hi, my name is John Doe. I would love to know more about your PCs")]

# Process conversation with streaming response
for chunk in graph.stream(
   {"messages": input_msg},  # Message content
   config,                   # Config params
   stream_mode="values"      # Stream format
):
   chunk["messages"][-1].pretty_print()


# User input 
input_messages = [HumanMessage(content="I last purchased the Latest generation Intel processor based PC and it worked really well. What are your prices on this models?")]

# Run the graph
for chunk in graph.stream({"messages": input_messages}, config, stream_mode="values"):
    chunk["messages"][-1].pretty_print()


