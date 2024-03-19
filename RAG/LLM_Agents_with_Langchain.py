## From https://towardsdatascience.com/intro-to-llm-agents-with-langchain-when-rag-is-not-enough-7d8c08145834

## Basic chain
from langchain import hub
from langchain.chains import SequentialChain

cot_step1 = hub.pull("rachnogstyle/nlw_jan24_cot_step1")
cot_step2 = hub.pull("rachnogstyle/nlw_jan24_cot_step2")
cot_step3 = hub.pull("rachnogstyle/nlw_jan24_cot_step3")
cot_step4 = hub.pull("rachnogstyle/nlw_jan24_cot_step4")

model = "gpt-3.5-turbo"

chain1 = LLMChain(
    llm=ChatOpenAI(temperature=0, model=model),
    prompt=cot_step1,
    output_key="solutions"
)

chain2 = LLMChain(
    llm=ChatOpenAI(temperature=0, model=model),
    prompt=cot_step2,
    output_key="review"
)

chain3 = LLMChain(
    llm=ChatOpenAI(temperature=0, model=model),
    prompt=cot_step3,
    output_key="deepen_thought_process"
)

chain4 = LLMChain(
    llm=ChatOpenAI(temperature=0, model=model),
    prompt=cot_step4,
    output_key="ranked_solutions"
)

overall_chain = SequentialChain(
    chains=[chain1, chain2, chain3, chain4],
    input_variables=["input", "perfect_factors"],
    output_variables=["ranked_solutions"],
    verbose=True
)

prompt = hub.pull("hwchase17/react")
prompt = hub.pull("hwchase17/self-ask-with-search")

### Memory ###
## Short-Term Memory ##
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [retriever_tool]
agent = create_openai_functions_agent(
    llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

message_history = ChatMessageHistory()
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


## Long-Term Memory ##
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

loader = WebBaseLoader("https://neurons-lab.com/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_documents(docs)
vector = FAISS.from_documents(documents, OpenAIEmbeddings())
retriever = vector.as_retriever()


## Tool ##
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain.tools.tavily_search import TavilySearchResults

search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
agent_chain = initialize_agent(
    [retriever_tool, tavily_tool],
    llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


## Custom Tool ##
from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool

@tool
def calculate_length_tool(a: str) -> int:
    """The function calculates the length of the input string."""
    return len(a)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)
agent_chain = initialize_agent(
    [retriever_tool, tavily_tool, calculate_length_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)


## Final step ##
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: message_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

