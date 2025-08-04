### From https://ai.plainenglish.io/building-agentic-rag-with-langgraph-mastering-adaptive-rag-for-production-c2c4578c836a
"""
1. Introduction
   Retrieval-Augmented Generation (RAG) has fundamentally changed how AI systems reason over external knowledge. 
   But traditional linear RAG pipelines are now insufficient for real-world, complex queries.
   As AI use cases scale, we are seeing the rise of intelligent, adaptive retrieval-generation systems 
   — chief among them is Adaptive RAG, which tailors its retrieval and generation strategy based on query complexity.

2. What Is Adaptive RAG?
   Adaptive RAG is an advanced evolution of RAG that integrates:
   -a. Query complexity classification
   -b. Active/self-corrective execution strategies
   It is built on the core insight: Not all queries are created equal.
   
   2.1 Types of Queries in Practice
       -a. Simple queries 
           Example: “What is the capital of France?” → Easily answerable via the LLM’s parametric knowledge.
       -b. Multi-hop reasoning queries
           Example: “When did the people who captured Malakoff come to the region where Philipsburg is located?”
           → Requires 4+ reasoning steps → Retrieval + intermediate generation needed.
   2.2 Three Approaches to RAG
       -a. Single-Step Retrieval
           -1. Retrieves documents once, then generates.
           -2. Efficient for simple queries.
           -3. Fails for multi-hop reasoning.
       -b. Multi-Step Retrieval
           -1. Iteratively retrieves + generates intermediate steps.
           -2. Great for complex reasoning.
           -3. Overkill for simple queries → high compute cost.
       -c. Adaptive RAG
           -1. Uses a trained query complexity classifier.
           -2. Dynamically routes each query to:
               -1) No retrieval
               -2) Single-step retrieval
               -3) Multi-hop iterative approach
                   → Balances efficiency and reasoning capability.

3. Adaptive RAG Workflow
   Built on LangGraph, Adaptive RAG uses a state-machine-like architecture to orchestrate complex decision-making. Its workflow includes:
   -a. Query Routing & Classification
       -1. A trained classifier analyzes the query and determines:
           -1) Whether retrieval is necessary
           -2) What level of retrieval is appropriate
       -2. It enables:
           -1) No-retrieval (for parametric answers)
           -2) Single-step retrieval (lightweight reasoning)
           -3) Multi-hop retrieval (deep reasoning)
   -b. Dynamic Knowledge Acquisition
       The system chooses one of the following based on routing:
       -1. Index-based retrieval: If the internal database suffices
       -2. Web search: If fresh or missing info is needed
       -3. No retrieval: For questions answerable by the LLM alone
   -c. Multi-Stage Quality Assurance
       Robust safeguards ensure answer reliability:
       -1. Document relevance scoring: Filters retrieved documents
       -2. Hallucination detection: Ensures generated answer is evidence-grounded
       -3. Answer quality check: Confirms final output answers the original query
"""

-----------------------------------------------------------------------------------------------------------------------------------
####### Environment Setup with UV
curl -LsSf <https://astral.sh/uv/install.sh> | sh
mkdir building-adaptive-rag
cd building-adaptive-rag

uv venv --python 3.10
source .venv/bin/activate  # On Windows: .venv\\Scripts\\activate

####### requirements.txt
beautifulsoup4
langchain-community
tiktoken
langchainhub
langchain
langgraph
tavily-python
langchain-openai
python-dotenv
black
isort
pytest
langchain-chroma
langchain-tavily==0.1.5
langchain_aws
langchain_google_genai

uv pip install -r requirements.txt

-----------------------------------------------------------------------------------------------------------------------------------
####### graph/state.py
from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]

####### graph/consts.py
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"

####### graph/chains/router.py
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from model import llm_model

class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "websearch"] = Field(
        ...,
        description="Given a user question choose to route it to web search or a vectorstore.",
    )

llm = llm_model

structured_llm_router = llm.with_structured_output(RouteQuery)

system = """You are an expert at routing a user question to a vectorstore or web search.
The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vectorstore for questions on these topics. For all else, use web-search."""
route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = route_prompt | structured_llm_router

####### graph/chains/retrieval_grader.py
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from model import llm_model

llm = llm_model

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader

####### graph/chains/hallucination_grader.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from model import llm_model

llm =  llm_model

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: bool = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucinations)

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
     Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader

####### graph/chains/answer_grader.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from model import llm_model


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

llm =  llm_model

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are a grader assessing whether an answer addresses / resolves a question \n 
     Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader

####### graph/chains/generation.py
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from model import llm_model

llm = llm_model

prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()

####### graph/nodes/retrieve.py
from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print("---RETRIEVE---")
    question = state["question"]

    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

####### graph/nodes/grade_documents.py
from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}
  
####### graph/nodes/web_search.py
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    
    # Initialize documents - this was the missing part!
    documents = state.get("documents", [])  # Get existing documents or empty list
    
    tavily_results = web_search_tool.invoke({"query": question})["results"]
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavily_results]
    )
    web_results = Document(page_content=joined_tavily_result)
    
    # Add web results to existing documents (or create new list if documents was empty)
    if documents:
        documents.append(web_results)
    else:
        documents = [web_results]
    
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})

####### graph/nodes/generate.py
from typing import Any, Dict

from graph.chains.generation import generation_chain
from graph.state import GraphState


def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

####### graph/graph.py
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.router import RouteQuery, question_router
from graph.consts import GENERATE, GRADE_DOCUMENTS, RETRIEVE, WEBSEARCH
from graph.nodes import generate, grade_documents, retrieve, web_search
from graph.state import GraphState

load_dotenv()

def decide_to_generate(state):
    print("---ASSESS GRADED DOCUMENTS---")

    if state["web_search"]:
        print(
            "---DECISION: NOT ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return WEBSEARCH
    else:
        print("---DECISION: GENERATE---")
        return GENERATE

def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


def route_question(state: GraphState) -> str:
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """
     
    print("---ROUTE QUESTION---")
    question = state["question"]
    source: RouteQuery = question_router.invoke({"question": question})

    if source.datasource == WEBSEARCH:
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return WEBSEARCH
    elif source.datasource == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return RETRIEVE


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_conditional_entry_point(
    route_question,
    {
        WEBSEARCH: WEBSEARCH,
        RETRIEVE: RETRIEVE,
    },
)

# workflow.set_entry_point(RETRIEVE)

workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {
        WEBSEARCH: WEBSEARCH,
        GENERATE: GENERATE,
    },
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {
        "not supported": GENERATE,
        "useful": END,
        "not useful": WEBSEARCH,
    },
)
workflow.add_edge(WEBSEARCH, GENERATE)
workflow.add_edge(GENERATE, END)

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="graph.png")

####### graph/chains/tests/test_chains.py
from pprint import pprint
import pytest

from dotenv import load_dotenv

load_dotenv()


from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucinations,
                                               hallucination_grader)
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "yes"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    # Skip test if no documents are returned
    # if len(docs) == 0:
    #     pytest.skip(f"No documents returned for query: {question}")

    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
  
-----------------------------------------------------------------------------------------------------------------------------------
####### model.py
from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI
# from langchain_aws import ChatBedrock
from langchain_google_genai import ChatGoogleGenerativeAI  

# from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# chat model
# llm_model = ChatOpenAI(temperature=0)
# llm_model =  ChatBedrock(model_id="anthropic.claude-sonnet-4-20250514-v1:0", region_name="us-west-2", temperature=0)   
llm_model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",                     
    temperature=0,                                
)

# embedding model
embed_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

####### ingestion.py
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from model import embed_model

load_dotenv()

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

doc_splits = text_splitter.split_documents(docs_list)

embed = embed_model

# Create vector store with documents
vectorstore = Chroma.from_documents(
    documents=doc_splits,
    collection_name="rag-chroma",
    embedding=embed,
    persist_directory="./.chroma",
)

# Create retriever
retriever = vectorstore.as_retriever()

-----------------------------------------------------------------------------------------------------------------------------------
####### main.py
from dotenv import load_dotenv

load_dotenv()

from graph.graph import app

def format_response(result):
    """Format the response from the graph for better readability"""
    if isinstance(result, dict) and "generation" in result:
        return result["generation"]
    elif isinstance(result, dict) and "answer" in result:
        return result["answer"]
    else:
        # Fallback to string representation
        return str(result)


def main():
    print("=" * 60)
    print("🤖 Advanced RAG Chatbot")
    print("=" * 60)
    print("Welcome! Ask me anything or type 'quit', 'exit', or 'bye' to stop.")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            user_question = input("\n💬 You: ").strip()
            
            # Check for exit commands
            if user_question.lower() in ['quit', 'exit', 'bye', 'q']:
                print("\n👋 Goodbye! Thanks for chatting!")
                break
            
            # Skip empty inputs
            if not user_question:
                print("Please enter a question.")
                continue
            
            # Show processing indicator
            print("\n🤔 Bot: Thinking...")
            
            # Process the question through the graph
            result = app.invoke(input={"question": user_question})
            
            # Format and display the response
            response = format_response(result)
            print(f"\n🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for chatting!")
            break
        except Exception as e:
            print(f"\n❌ Sorry, I encountered an error: {str(e)}")
            print("Please try asking your question again.")


if __name__ == "__main__":
    main()
