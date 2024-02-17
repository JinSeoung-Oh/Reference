# From https://towardsdatascience.com/using-langchain-react-agents-for-answering-multi-hop-questions-in-rag-systems-893208c1847e

from langchain_openai import AzureChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import os
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.storage._lc_store import create_kv_docstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import RetrievalQA

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_DEPLOYMENT_NAME = "gpt-35-turbo-16k"
OPENAI_DEPLOYMENT_ENDPOINT = "https://<???>.openai.azure.com/"
OPENAI_DEPLOYMENT_VERSION = "2023-12-01-preview"
OPENAI_MODEL_NAME = "gpt-35-turbo-16k"

OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME = "text-embedding-ada"
OPENAI_ADA_EMBEDDING_MODEL_NAME = "text-embedding-ada-002"

llm = AzureChatOpenAI(
    deployment_name=OPENAI_DEPLOYMENT_NAME,
    model_name=OPENAI_MODEL_NAME,
    openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
    openai_api_version=OPENAI_DEPLOYMENT_VERSION,
    openai_api_key=OPENAI_API_KEY,
    openai_api_type="azure",
    temperature=0.1,
)

embeddings = OpenAIEmbeddings(
    deployment=OPENAI_ADA_EMBEDDING_DEPLOYMENT_NAME,
    model=OPENAI_ADA_EMBEDDING_MODEL_NAME,
    openai_api_base=OPENAI_DEPLOYMENT_ENDPOINT,
    openai_api_type="azure",
    chunk_size=1,
    openai_api_key=OPENAI_API_KEY,
    openai_api_version=OPENAI_DEPLOYMENT_VERSION,
)

loader = TextLoader("Your txt file directory")
documents = loader.load()

persist_directory = "local_vectorstore"
collection_name = "hrpolicy"
PROJECT_ROOT = "...." #insert your project root directory name here

vectorstore = Chroma(
    persist_directory=os.path.join(PROJECT_ROOT, "data", persist_directory),
    collection_name=collection_name,
    embedding_function=embeddings,
)

local_store = "local_docstore"
local_store = LocalFileStore(os.path.join(PROJECT_ROOT, "data", local_store))
docstore = create_kv_docstore(local_store)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

#### ReAct Agent
### Defining the tools for the agent
##  In this part, ReAct Agent and retriever are connected
from langchain.tools.retriever import create_retriever_tool

tool_search = create_retriever_tool(
    retriever=retriever,
    name="search_hr_policy",
    description="Searches and returns excerpts from the HR policy.",
)

### Creating a ReAct agent
from langchain import hub
prompt = hub.pull("hwchase17/react")

from langchain.agents import AgentExecutor, create_react_agent

react_agent = create_react_agent(llm, [tool_search], prompt)
agent_executor = AgentExecutor(
agent=react_agent, 
tools=[tool],
verbose=True,
handle_parsing_errors=True,
max_iterations = 5 # useful when agent is stuck in a loop
)

## testing
query = "Which country has the highest budget?"
agent_executor.invoke({"input": query})
