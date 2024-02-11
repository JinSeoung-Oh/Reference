# From https://medium.com/ai-advances/unveiling-the-future-of-ai-collaboration-with-langgraph-embracing-multi-agent-workflows-89a909ddd455

!pip install langchain transformers openai langgraph  langchain_openai tavily-python -q

from getpass import getpass
import os

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass(f"Please provide your {var}")

# Confirming required API keys. _set_if_undefined is not function, It mean you have to build for this
_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("LANGCHAIN_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

# Configuration for LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "blog_supervisor_dev"

LLM_SMART_MODEL = "gpt-3.5-turbo-16k"

from langchain_core.runnables import Runnable
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_community.chat_models import ChatOpenAI
from typing import TypedDict


class AgentDescription(TypedDict):
    name: str
    description: str
    
def create_agent(
        llm: ChatOpenAI,
        tools: list,
        system_prompt: str,
) -> AgentExecutor:
    system_prompt += "\nWork autonomously according to your specialty, using the tools available to you."
    " Do not ask for clarification."
    " Your other team members (and other teams) will collaborate with you with their own specialties."
    " You are chosen for a reason! You are one of the following team members: {team_members}."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)


def create_team_supervisor(
        llm: ChatOpenAI,
        system_prompt: str,
        members: list[AgentDescription]
) -> Runnable:
    member_names = [member["name"] for member in members]
    team_members = []
    for member in members:
        team_members.append(f"name: {member['name']}\ndescription: {member['description']}")
    options = ["FINISH"] + member_names
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of option: {options}",
            ),
        ]
    ).partial(options=str(options), team_members="\n\n".join(team_members))
    return (
            prompt
            | llm.bind_functions(functions=[function_def], function_call="route")
            | JsonOutputFunctionsParser()
    )


from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

llm = ChatOpenAI(model_name=LLM_SMART_MODEL)
tavily_tool = TavilySearchResults(max_results=5)


@tool
def scrape_webpages(urls: list[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )


def create_research_agent() -> Runnable:
    prompt = "You are a research assistant who can search for up-to-date info using the tavily search engine."
    return create_agent(llm, [tavily_tool, scrape_webpages], prompt)


import operator
from typing import Annotated
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import BaseMessage, AIMessage

# Definition of each node
RESEARCH_NODE = "research"
QUALITY_ASSURANCE_NODE = "quality_assurance"
WRITER_NODE = "writer"
SUPERVISOR_NODE = "supervisor"

# Definition of team members
team_members = [
    {"name": RESEARCH_NODE,
     "description": "Search the web for necessary information and write articles as requested by users."},
    {"name": QUALITY_ASSURANCE_NODE,
     "description": f"Check that the quality of the article meets the criteria. If not, return feedback. The acceptability of deliverables is always determined by {QUALITY_ASSURANCE_NODE}."},
    {"name": WRITER_NODE,
     "description": "Write articles that even the world's best AI checkers would judge as written by a human."},
]

# Definition of the top-level state
class TopState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    output: str
    next: str

# Function to create the Language Model chain
def create_llm_chain(llm: ChatOpenAI, system_message: str) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_message), ("user", "{message}")]
    )
    chain = {"message": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    return chain

# Definition of the Researcher node
def research_node(state: TopState) -> dict:
    last_message = state["messages"][-1]
    result = create_research_agent().invoke({"messages": [last_message]})
    return {
        "output": result["output"],
        "messages": [AIMessage(content=result["output"])]
    }

# Definition of the Writer node
def writer_node(state: TopState) -> dict:
    system_message = """You are a writing specialist.
    You can write sentences that even the world's best AI checkers would judge as written by a human."""
    latest_message = state["messages"][-1].content
    chain = create_llm_chain(llm, system_message)
    result = chain.invoke(latest_message)
    return {
        "output": result,
        "messages": [AIMessage(content=result)]
    }

# Definition of the Quality Assurance node
def qa_node(state: TopState) -> dict:
    system_message = """You are a specialist in inspecting the quality of articles. 
    Inspect the article to see if it meets the following requirements:
    - The article is written in English.
    - The writing style MUST be such that the AI sentence checker determines that it was written by a HUMAN.
    - The article MUST be written in a way that is easy to understand.
    - The article MUST meet the user's requirements.
    If the evaluation criteria are met, write "Acceptable".
    In addition, write the reason why you judged that the evaluation criteria are met.
    If the evaluation criteria are not met, write "Not Acceptable".
    In addition, provide feedback on what needs to be done to meet the evaluation criteria.
    DO NOT make excuses such as "I can't make a decision because I am an AI".
    The quality of your articles is relevant to your career.
    Please be as rigorous as possible in your inspections and make sure that your feedback is helpful in making corrections."""
    latest_output = state["output"]
    chain = create_llm_chain(llm, system_message)
    result = chain.invoke(latest_output)
    result_with_original_article = f"feedback: {result}\n\n-----\n{latest_output}"
    return {"messages": [AIMessage(content=result_with_original_article)]}

# Definition of the Supervisor node
def supervisor_node(state: TopState) -> Runnable:
    prompt = """You are a supervisor tasked with managing a conversation between the following teams:
    {team_members}
    Given the following user request, respond with the worker to act next. 
    Each worker will perform a task and respond with their results and status.
    When finished, respond with FINISH."""
    return create_team_supervisor(llm, prompt, team_members)

from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

graph = StateGraph(TopState)

graph.add_node(RESEARCH_NODE, research_node)
graph.add_node(QUALITY_ASSURANCE_NODE, qa_node)
graph.add_node(WRITER_NODE, writer_node)
graph.add_node(SUPERVISOR_NODE, supervisor_node)

graph.add_edge(RESEARCH_NODE, SUPERVISOR_NODE)
graph.add_edge(QUALITY_ASSURANCE_NODE, SUPERVISOR_NODE)
graph.add_edge(WRITER_NODE, SUPERVISOR_NODE)
graph.add_conditional_edges(
    SUPERVISOR_NODE,
    lambda x: x["next"],
    {
        RESEARCH_NODE: RESEARCH_NODE,
        QUALITY_ASSURANCE_NODE: QUALITY_ASSURANCE_NODE,
        WRITER_NODE: WRITER_NODE,
        "FINISH": END,
    }
)

graph.set_entry_point(SUPERVISOR_NODE)
blog_writer = {"messages": lambda x: [HumanMessage(content=x)]} | graph.compile()


query = """Write an explanatory article in English for business professionals about the following news. Write in a way that the AI sentence checker does not determine it to be written by AI.
https://techcrunch.com/2024/01/25/kids-spent-60-more-time-on-tiktok-than-youtube-last-year-20-tried-openais-chatgpt/
"""

latest_output = ""
for s in blog_writer.stream(query, {"recursion_limit": 100}):
    if "__end__" not in s:
        print(s)
        print("---")
        writing_output = (
                s.get(RESEARCH_NODE, {}).get("output") or
                s.get(WRITER_NODE, {}).get("output")
        )
        if writing_output:
            latest_output = writing_output
