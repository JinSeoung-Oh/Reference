### From https://medium.com/@nicoladisabato_19197/building-rag-research-multi-agent-with-langgraph-1bd47acac69f

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field
from pydantic import BaseModel, Field
from dataclasses import dataclass, field
from typing import Annotated, Literal, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from utils.utils import reduce_docs

logger.info("Starting document processing.")
converter = DocumentConverter()
markdown_document = converter.convert(source).document.export_to_markdown()

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2")
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
docs_list = markdown_splitter.split_text(markdown_document)

embd = OpenAIEmbeddings()

vectorstore_from_documents = Chroma.from_documents(
    documents=docs_list,
    collection_name="rag-chroma-google-v1",
    embedding=embd,
    persist_directory='db_vector'
)

class Router(TypedDict):
    """Classify user query."""

    logic: str
    type: Literal["more-info", "environmental", "general"]

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, '1' or '0'"
    )

@dataclass(kw_only=True)
class InputState:
    """Represents the input state for the agent.

    This class defines the structure of the input state, which includes
    the messages exchanged between the user and the agent. It serves as
    a restricted version of the full State, providing a narrower interface
    to the outside world compared to what is maintained iternally.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    
    """Messages track the primary execution state of the agent.

    Typically accumulates a pattern of Human/AI/Human/AI messages.

    Returns:
        A new list of messages with the messages from `right` merged into `left`.
        If a message in `right` has the same ID as a message in `left`, the
        message from `right` will replace the message from `left`."""

# Primary agent state
@dataclass(kw_only=True)
class AgentState(InputState):
    """State of the retrieval graph / agent."""

    router: Router = field(default_factory=lambda: Router(type="general", logic=""))
    """The router's classification of the user's query."""
    steps: list[str] = field(default_factory=list)
    """A list of steps in the research plan."""
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""
    hallucination: GradeHallucinations = field(default_factory=lambda: GradeHallucinations(binary_score="0"))

async def analyze_and_route_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Router]:
    """Analyze the user's query and determine the appropriate routing.

    This function uses a language model to classify the user's query and decide how to route it
    within the conversation flow.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """
    model = ChatOpenAI(model=GPT_4o, temperature=TEMPERATURE, streaming=True)
    messages = [
        {"role": "system", "content": ROUTER_SYSTEM_PROMPT}
    ] + state.messages
    logging.info("---ANALYZE AND ROUTE QUERY---")
    response = cast(
        Router, await model.with_structured_output(Router).ainvoke(messages)
    )
    return {"router": response}


def route_query(
    state: AgentState,
) -> Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]:
    """Determine the next step based on the query classification.

    Args:
        state (AgentState): The current state of the agent, including the router's classification.

    Returns:
        Literal["create_research_plan", "ask_for_more_info", "respond_to_general_query"]: The next step to take.

    Raises:
        ValueError: If an unknown router type is encountered.
    """
    _type = state.router["type"]
    if _type == "environmental":
        return "create_research_plan"
    elif _type == "more-info":
        return "ask_for_more_info"
    elif _type == "general":
        return "respond_to_general_query"
    else:
        raise ValueError(f"Unknown router type {_type}")

async def ask_for_more_info(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response asking the user for more information.

    This node is called when the router determines that more information is needed from the user.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = MORE_INFO_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}



async def respond_to_general_query(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a response to a general query not related to environmental.

    This node is called when the router classifies the query as a general question.

    Args:
        state (AgentState): The current state of the agent, including conversation history and router logic.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = GENERAL_SYSTEM_PROMPT.format(
        logic=state.router["logic"]
    )
    logging.info("---RESPONSE GENERATION---")
    messages = [{"role": "system", "content": system_prompt}] + state.messages
    response = await model.ainvoke(messages)
    return {"messages": [response]}

async def create_research_plan(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[str] | str]:
    """Create a step-by-step research plan for answering a environmental-related query.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used to generate the plan.

    Returns:
        dict[str, list[str]]: A dictionary with a 'steps' key containing the list of research steps.
    """

    class Plan(TypedDict):
        """Generate research plan."""

        steps: list[str]

    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    messages = [
        {"role": "system", "content": RESEARCH_PLAN_SYSTEM_PROMPT}
    ] + state.messages
    logging.info("---PLAN GENERATION---")
    response = cast(Plan, await model.with_structured_output(Plan).ainvoke(messages))
    return {"steps": response["steps"], "documents": "delete"}

async def conduct_research(state: AgentState) -> dict[str, Any]:
    """Execute the first step of the research plan.

    This function takes the first step from the research plan and uses it to conduct research.

    Args:
        state (AgentState): The current state of the agent, including the research plan steps.

    Returns:
        dict[str, list[str]]: A dictionary with 'documents' containing the research results and
                              'steps' containing the remaining research steps.

    Behavior:
        - Invokes the researcher_graph with the first step of the research plan.
        - Updates the state with the retrieved documents and removes the completed step.
    """
    result = await researcher_graph.ainvoke({"question": state.steps[0]}) #graph call directly
    docs = result["documents"]
    step = state.steps[0]
    logging.info(f"\n{len(docs)} documents retrieved in total for the step: {step}.")
    return {"documents": result["documents"], "steps": state.steps[1:]}

------------------------------------------------------------------------------------------------------
# Researcher subgraph building
"""States for the researcher subgraph.

This module defines the state structures used in the researcher subgraph.
"""

from dataclasses import dataclass, field
from typing import Annotated
from langchain_core.documents import Document
from utils.utils import reduce_docs
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_community.llms import Cohere

@dataclass(kw_only=True)
class QueryState:
    """Private state for the retrieve_documents node in the researcher graph."""
    query: str

@dataclass(kw_only=True)
class ResearcherState:
    """State of the researcher graph / agent."""
    question: str
    """A step in the research plan generated by the retriever agent."""
    queries: list[str] = field(default_factory=list)
    """A list of search queries based on the question that the researcher generates."""
    documents: Annotated[list[Document], reduce_docs] = field(default_factory=list)
    """Populated by the retriever. This is a list of documents that the agent can reference."""

async def generate_queries(
    state: ResearcherState, *, config: RunnableConfig
) -> dict[str, list[str]]:
    """Generate search queries based on the question (a step in the research plan).

    This function uses a language model to generate diverse search queries to help answer the question.

    Args:
        state (ResearcherState): The current state of the researcher, including the user's question.
        config (RunnableConfig): Configuration with the model used to generate queries.

    Returns:
        dict[str, list[str]]: A dictionary with a 'queries' key containing the list of generated search queries.
    """

    class Response(TypedDict):
        queries: list[str]

    logger.info("---GENERATE QUERIES---")
    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0)
    messages = [
        {"role": "system", "content": GENERATE_QUERIES_SYSTEM_PROMPT},
        {"role": "human", "content": state.question},
    ]
    response = cast(Response, await model.with_structured_output(Response).ainvoke(messages))
    queries = response["queries"]
    queries.append(state.question)
    logger.info(f"Queries: {queries}")
    return {"queries": response["queries"]}

def _setup_vectorstore() -> Chroma:
    """
    Set up and return the Chroma vector store instance.
    """
    embeddings = OpenAIEmbeddings()
    return Chroma(
        collection_name=VECTORSTORE_COLLECTION,
        embedding_function=embeddings,
        persist_directory=VECTORSTORE_DIRECTORY
    )

# Create base retrievers
retriever_bm25 = BM25Retriever.from_documents(documents, search_kwargs={"k": TOP_K})
retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": TOP_K})

ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever_vanilla, retriever_mmr, retriever_bm25],
        weights=ENSEMBLE_WEIGHTS,
    )

# Set up Cohere re-ranking
compressor = CohereRerank(top_n=2, model="rerank-english-v3.0")

# Build compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble_retriever,
)

compression_retriever.invoke(
    "Retrieve the data center PUE efficiency in Dublin in 2019"
)

async def retrieve_and_rerank_documents(
    state: QueryState, *, config: RunnableConfig
) -> dict[str, list[Document]]:
    """Retrieve documents based on a given query.

    This function uses a retriever to fetch relevant documents for a given query.

    Args:
        state (QueryState): The current state containing the query string.
        config (RunnableConfig): Configuration with the retriever used to fetch documents.

    Returns:
        dict[str, list[Document]]: A dictionary with a 'documents' key containing the list of retrieved documents.
    """
    logger.info("---RETRIEVING DOCUMENTS---")
    logger.info(f"Query for the retrieval process: {state.query}")

    response = compression_retriever.invoke(state.query)

    return {"documents": response}

def check_finished(state: AgentState) -> Literal["respond", "conduct_research"]:
    """Determine if the research process is complete or if more research is needed.

    This function checks if there are any remaining steps in the research plan:
        - If there are, route back to the `conduct_research` node
        - Otherwise, route to the `respond` node

    Args:
        state (AgentState): The current state of the agent, including the remaining research steps.

    Returns:
        Literal["respond", "conduct_research"]: The next step to take based on whether research is complete.
    """
    if len(state.steps or []) > 0:
        return "conduct_research"
    else:
        return "respond"

async def respond(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, list[BaseMessage]]:
    """Generate a final response to the user's query based on the conducted research.

    This function formulates a comprehensive answer using the conversation history and the documents retrieved by the researcher.

    Args:
        state (AgentState): The current state of the agent, including retrieved documents and conversation history.
        config (RunnableConfig): Configuration with the model used to respond.

    Returns:
        dict[str, list[str]]: A dictionary with a 'messages' key containing the generated response.
    """
    print("--- RESPONSE GENERATION STEP ---")
    model = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)
    context = format_docs(state.documents)
    prompt = RESPONSE_SYSTEM_PROMPT.format(context=context)
    messages = [{"role": "system", "content": prompt}] + state.messages
    response = await model.ainvoke(messages)

    return {"messages": [response]}

async def check_hallucinations(
    state: AgentState, *, config: RunnableConfig
) -> dict[str, Any]:
    """Analyze the user's query and checks if the response is supported by the set of facts based on the document retrieved,
    providing a binary score result.

    This function uses a language model to analyze the user's query and gives a binary score result.

    Args:
        state (AgentState): The current state of the agent, including conversation history.
        config (RunnableConfig): Configuration with the model used for query analysis.

    Returns:
        dict[str, Router]: A dictionary containing the 'router' key with the classification result (classification type and logic).
    """

    model = ChatOpenAI(model=GPT_4o_MINI, temperature=TEMPERATURE, streaming=True)
    system_prompt = CHECK_HALLUCINATIONS.format(
        documents=state.documents,
        generation=state.messages[-1]
    )

    messages = [
        {"role": "system", "content": system_prompt}
    ] + state.messages
    logging.info("---CHECK HALLUCINATIONS---")
    response = cast(GradeHallucinations, await model.with_structured_output(GradeHallucinations).ainvoke(messages))
    
    return {"hallucination": response}

def human_approval(
    state: AgentState,
):
    _binary_score = state.hallucination.binary_score
    if _binary_score == "1":
        return "END"
    else:
        retry_generation = interrupt(
        {
            "question": "Is this correct?",
            "llm_output": state.messages[-1]
        })

        if retry_generation == "y":
            print("voglio continuare")
            return "respond"
        else:
            return "END"

------------------------------------------------------------------------------------------------
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

builder = StateGraph(AgentState, input=InputState)
builder.add_node(analyze_and_route_query)
builder.add_edge(START, "analyze_and_route_query")
builder.add_conditional_edges("analyze_and_route_query", route_query)
builder.add_node(create_research_plan)
builder.add_node(ask_for_more_info)
builder.add_node(respond_to_general_query)
builder.add_node(conduct_research)
builder.add_node("respond", respond)
builder.add_node(check_hallucinations)
builder.add_conditional_edges("check_hallucinations", human_approval, {"END": END, "respond": "respond"})
builder.add_edge("create_research_plan", "conduct_research")
builder.add_conditional_edges("conduct_research", check_finished)
builder.add_edge("respond", "check_hallucinations")

graph = builder.compile(checkpointer=checkpointer)

------------------------------------------------------------------------------------------------
from subgraph.graph_states import ResearcherState
from main_graph.graph_states import AgentState
from utils.utils import config, new_uuid
from subgraph.graph_builder import researcher_graph
from main_graph.graph_builder import InputState, graph
from langgraph.types import Command
import asyncio
import uuid

import asyncio
import time
import builtins

thread = {"configurable": {"thread_id": new_uuid()}}

async def process_query(query):
    inputState = InputState(messages=query)

    async for c, metadata in graph.astream(input=inputState, stream_mode="messages", config=thread):
        if c.additional_kwargs.get("tool_calls"):
            print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="", flush=True)
        if c.content:
            time.sleep(0.05)
            print(c.content, end="", flush=True)

    if len(graph.get_state(thread)[-1]) > 0:
        if len(graph.get_state(thread)[-1][0].interrupts) > 0:
            response = input("\nThe response may contain uncertain information. Retry the generation? If yes, press 'y': ")
            if response.lower() == 'y':
                async for c, metadata in graph.astream(Command(resume=response), stream_mode="messages", config=thread):
                    if c.additional_kwargs.get("tool_calls"):
                        print(c.additional_kwargs.get("tool_calls")[0]["function"].get("arguments"), end="")
                    if c.content:
                        time.sleep(0.05)
                        print(c.content, end="", flush=True)


async def main():
    input = builtins.input
    print("Enter your query (type '-q' to quit):")
    while True:
        query = input("> ")
        if query.strip().lower() == "-q":
            print("Exiting...")
            break
        await process_query(query)


if __name__ == "__main__":
    asyncio.run(main())


graph.astream(Command(resume=response), stream_mode="messages", config=thread)



builder = StateGraph(ResearcherState)
builder.add_node(generate_queries)
builder.add_node(retrieve_and_rerank_documents)
builder.add_edge(START, "generate_queries")
builder.add_conditional_edges(
    "generate_queries",
    retrieve_in_parallel,  # type: ignore
    path_map=["retrieve_and_rerank_documents"],
)
builder.add_edge("retrieve_and_rerank_documents", END)
researcher_graph = builder.compile()











