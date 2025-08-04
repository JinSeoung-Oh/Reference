### From https://levelup.gitconnected.com/optimizing-langchain-ai-agents-with-contextual-engineering-0914d84601f3

""""
1. What is Context Engineering?
   Context Engineering refers to the design of what information is placed inside an LLM’s context window (equivalent to RAM) 
   and how it is managed.
   It mirrors how humans decide what to remember, what to summarize, and what to retain temporarily.
  
   Since LLMs have limited context windows, inserting too much information can cause:
   -a. Context Poisoning: Incorrect or hallucinated information accumulates
   -b. Context Distraction: Irrelevant info lowers model accuracy
   -c. Context Confusion: The model misinterprets context
   -d. Context Clash: Conflicting pieces of information confuse the output

   To solve these issues, 4 key strategies are used:
       Write, Select, Compress, Isolate

2. Write: Crafting and Recording Context
   -a. Scratchpad: Short-term memory mechanism to store notes or interim data
   -b. In LangGraph, this is implemented using the state object, which can be shared across nodes
   -c. Checkpointing enables saving session state at various points for reliable recall

   LangGraph structural components:
   -a. StateGraph: A DAG (Directed Acyclic Graph)-based workflow where each node receives and updates a state
   -b. TypedDict / Pydantic: Used to define and fix the structure of the state object

3. Select: Picking Only the Necessary Context
   Selection strategy controls what part of the state is exposed to the LLM at each step.

   -a. Example:
       -1. Selecting state["joke"] only and using a prompt like “Make this joke funnier” instead of sending the full state
   When combined with Memory Store, previously generated information (like a prior joke) can be reused.

   -b. Key Principle:
       Don’t dump the whole state — carefully select and expose only what's relevant to the current step.

4. Memory: Maintaining Context Across Sessions
   -a. Short-term memory: Maintained via state and checkpoint within a session
   -b. Long-term memory: Stored in InMemoryStore or external databases across sessions
   -c. Usage pattern:
       -1. Save generated content (e.g., jokes) from a past session
       -2. Retrieve it in future sessions using prompts like:
       | “Write a joke about cats that’s different from this: {prior_joke}”
   LangGraph supports various backends via its BaseStore interface.

5. RAG + Tool Orchestration: Intelligent Tool Selection via Retrieval
   -a. RAG (Retrieval-Augmented Generation) process:
       -1. External documents are chunked
       -2. Embedded into vectors
       -3. Stored in a VectorStore
       -4. Retrieved based on semantic similarity
   -b. Tool Loadout Strategy:
       As the number of tools grows, selection becomes harder.
       → Apply RAG over tool descriptions too, so only the most relevant tools are exposed to the agent.
   -c. langgraph-bigtool:
       -1. Embeds tool descriptions
       -2. Performs semantic filtering before presenting them to the LLM
   -d. Benefit: Up to 3x higher tool selection accuracy

6. Compress: Summarizing Context to Save Tokens
   -a. Summarization strategy is used when:
       -1. Conversations or tool outputs become too long
       -2. Token usage nears context window limits
   -b. Examples:
       -1. Claude uses auto-compact at 95% token usage
       -2. LangGraph allows:
           -1) summary nodes that summarize the full interaction
           -2) tool_node_with_summarization to summarize tool outputs immediately after use
   -c. Effect:
        → Up to 50% reduction in token usage, faster and more cost-efficient execution

7. Isolate: Separating Context to Prevent Confusion
   -a. Multi-Agent Systems:
       -1. Each agent operates with its own context window and specific role
       -2. Supervisor agent routes tasks to sub-agents (e.g., math_expert, research_expert)
       -3. Anthropic's paper shows this increases performance by 90.2%
   -b. Sandbox Execution:  
       -1. Instead of calling tools directly, the LLM writes Python code that is executed in a sandbox
       -2. Keeps heavy data (image/audio) out of token context
       -3. LangChain Sandbox uses Pyodide (WASM Python runtime) for secure execution
   -c. State Object Field Isolation:
       -1. Fields like messages, memory, tool_results, etc. are separated
       -2. Each is selectively exposed to the LLM as needed
       -3. Example schema:
           State(messages=..., tools=..., summary=...)

8. Final Summary
   Strategy	| Purpose	| LangGraph Support
   Write	| Record to scratchpad memory	| state object, checkpointing
   Select	| Expose only relevant information	| state field selection, memory lookup
   Compress	| Reduce context length for efficiency	| summary_node, tool output summarizer
   Isolate	| Prevent role/context clashes	| sub-agents, sandbox, state schema
""""
from typing import TypedDict
from rich.console import Console
from rich.pretty import pprint

console = Console()

class State(TypedDict):
    topic: str
    joke: str

######### Scratchpad & Short-term Memory
from langgraph.graph import StateGraph, START, END

def generate_joke(state: State) -> dict[str, str]:
    topic = state["topic"]
    print(f"Generating a joke about: {topic}")
    msg = llm.invoke(f"Write a short joke about {topic}")
    return {"joke": msg.content}

workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)
chain = workflow.compile()

joke_generator_state = chain.invoke({"topic": "cats"})
pprint(joke_generator_state)

######### Long-Term Memory (InMemoryStore)
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
namespace = ("rlm", "joke_generator")

store.put(
    namespace,
    "last_joke",
    {"joke": joke_generator_state["joke"]}
)

stored_items = list(store.search(namespace))
pprint(stored_items)

######### Memory-Aware Workflow
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.base import BaseStore

checkpointer = InMemorySaver()
memory_store = InMemoryStore()

def generate_joke(state: State, store: BaseStore) -> dict[str, str]:
    existing = store.get(namespace, "last_joke")
    prior_joke_text = existing.value["joke"] if existing else "None"

    prompt = f"Write a short joke about {state['topic']}, but make it different from: {prior_joke_text}"
    msg = llm.invoke(prompt)

    store.put(namespace, "last_joke", {"joke": msg.content})
    return {"joke": msg.content}

workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", END)
chain = workflow.compile(checkpointer=checkpointer, store=memory_store)

config = {"configurable": {"thread_id": "1"}}
state = chain.invoke({"topic": "cats"}, config)

######### Select
def improve_joke(state: State) -> dict[str, str]:
    print(f"Initial joke: {state['joke']}")
    msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['joke']}")
    return {"improved_joke": msg.content}

workflow = StateGraph(State)
workflow.add_node("generate_joke", generate_joke)
workflow.add_node("improve_joke", improve_joke)
workflow.add_edge(START, "generate_joke")
workflow.add_edge("generate_joke", "improve_joke")
workflow.add_edge("improve_joke", END)
chain = workflow.compile()

######### Tool Retriever
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.tools.retriever import create_retriever_tool

urls = [
    "https://lilianweng.github.io/posts/2025-05-01-thinking/",
    ...
]
docs = [WebBaseLoader(url).load() for url in urls]
docs = [doc for sub in docs for doc in sub]

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2000, chunk_overlap=50)
doc_splits = splitter.split_documents(docs)

vectorstore = InMemoryVectorStore.from_documents(doc_splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_blog_posts",
    "Search and return information about Lilian Weng blog posts."
)

######### Tool Calling Agent with LangGraph
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, ToolMessage
from typing_extensions import Literal

rag_prompt = """You are a helpful assistant retrieving info from Lilian Weng’s blogs..."""

def llm_call(state: MessagesState):
    messages = [SystemMessage(content=rag_prompt)] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

def tool_node(state: MessagesState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        obs = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=str(obs), tool_call_id=tool_call["id"]))
    return {"messages": result}

def should_continue(state: MessagesState) -> Literal["Action", END]:
    return "Action" if state["messages"][-1].tool_calls else END

graph = StateGraph(MessagesState)
graph.add_node("llm_call", llm_call)
graph.add_node("Action", tool_node)
graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call", should_continue, {"Action": "Action", END: END})
graph.add_edge("Action", "llm_call")
agent = graph.compile()

######### Summarization Node
class State(MessagesState):
    summary: str

def summary_node(state: MessagesState) -> dict:
    messages = [SystemMessage(content=summarization_prompt)] + state["messages"]
    result = llm.invoke(messages)
    return {"summary": result.content}

def should_continue(state: MessagesState) -> Literal["Action", "summary_node"]:
    return "Action" if state["messages"][-1].tool_calls else "summary_node"

graph = StateGraph(State)
graph.add_node("llm_call", llm_call)
graph.add_node("Action", tool_node)
graph.add_node("summary_node", summary_node)
graph.add_edge(START, "llm_call")
graph.add_conditional_edges("llm_call", should_continue, {"Action": "Action", "summary_node": "summary_node"})
graph.add_edge("Action", "llm_call")
graph.add_edge("summary_node", END)
agent = graph.compile()

######### Multi-Agent Supervisor Agent
from langgraph_supervisor import create_supervisor
from langgraph.prebuilt import create_react_agent

def add(a: float, b: float) -> float:
    return a + b

def multiply(a: float, b: float) -> float:
    return a * b

def web_search(query: str) -> str:
    return "FAANG headcounts..."

math_agent = create_react_agent(llm, [add, multiply], name="math_expert", prompt="You are a math expert.")
research_agent = create_react_agent(llm, [web_search], name="research_expert", prompt="You are a researcher.")

workflow = create_supervisor(
    [research_agent, math_agent],
    model=llm,
    prompt="You are a supervisor managing a math and research agent..."
)
app = workflow.compile()
-------------------------------------------------------------

from langchain_sandbox import PyodideSandboxTool
from langgraph.prebuilt import create_react_agent

tool = PyodideSandboxTool(allow_net=True)
agent = create_react_agent(llm, tools=[tool])

result = await agent.ainvoke({
    "messages": [{"role": "user", "content": "what's 5 + 7?"}]
})
