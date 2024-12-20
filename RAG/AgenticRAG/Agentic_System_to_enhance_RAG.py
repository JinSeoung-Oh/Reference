### From https://medium.com/the-ai-forum/building-an-agentic-system-to-enhance-rag-with-self-grading-and-web-search-capabilities-using-3f9a1d885730

## 1. System Architecture
-------------------------------------------------------
#*    -a. Document Processing and Indexing
loader = PyPDFLoader("path/to/document.pdf")
documents = loader.load()

# Text splitting
split_docs = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=50
).split_documents(documents)

# Vector store creation
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory
)

#*    -b. Agent Configuration
-------------------------------------------------------
groq_agent = Agent(
    groq_model,
    deps_type=Deps,
    retries=2,
    result_type=str,
    system_prompt="""You are a Helpful Assistant Proficient in 
    Answering concise, factual and to the point answers..."""
)

#*    -c. Retrieval Tool
-------------------------------------------------------
@groq_agent.tool
async def retriever_tool(ctx: RunContext[Deps], question: str) -> List[str]:
    load_vectorstore = Chroma(
        persist_directory=persist_directory, 
        embedding_function=embedding
    )
    docs = load_vectorstore.similarity_search(question, k=3)
    return [d.page_content for d in docs]

#*    -d. Web Search Integration
-------------------------------------------------------
@groq_agent.tool_plain
async def websearch_tool(question) -> str:
    tavily_client = TavilyClient()
    answer = tavily_client.qna_search(query=question)
    return answer

#################################################################################
## 2. Key Features
#*    -a. Self-Grading Mechanism
-------------------------------------------------------
{
    "Relevancy": 0.9,
    "Faithfulness": 0.95,
    "Context Quality": 0.85,
    "Needs Web Search": false,
    "Explanation": "Response directly addresses the question..."
}

#*    -b. Dynamic Context Augmentation
-------------------------------------------------------
if grades["Needs Web Search"]:
    web_results = await websearch_tool(query)
    # Augment context with web results

#*    -c. Structured Dependencies
-------------------------------------------------------
@dataclass
class Deps:
    question: str | None
    context: str | None

#################################################################################
## Code Implementation

%pip -q install pydantic-ai
%pip -q install nest_asyncio
%pip -q install devtools
%pip install 'pydantic-ai-slim[openai,groq,logfire]'
%pip install tavily-python
%pip install -qU langchain 
%pip install -qU langchain_community
%pip install -qU sentence_transformers
%pip install -qU langchain_huggingface
%pip install pypdf

from google.colab import userdata
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.models.groq import GroqModel
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dataclasses import dataclass
import nest_asyncio

os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY')
os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

@dataclass
class Deps:
    question:str |None
    context:str |None
  
openai_model = OpenAIModel('gpt-4o-mini')
groq_model = GroqModel("llama-3.3-70b-versatile")

loader = PyPDFLoader("/content/data/Fibromyalgia_Final.pdf")
documents = loader.load()

split_docs = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(documents)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Build Index
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embedding,
    persist_directory=persist_directory,
    collection_name="fibromyalgia"

nest_asyncio.apply()
groq_agent = Agent(groq_model,
                   deps_type=Deps,
                    retries=2,
                    result_type=str,
                   system_prompt=("You are a Helpful Assiatnt Profiocient in Answering concise,factful and to the point asnwers for questions asked based on the Context provided"
                   "You have to Use the `retrievre_tool' to get relevent context and generate response based on the context retrieved"
                   """You are a grading assistant. Evaluate the response based on:
        1. Relevancy to the question
        2. Faithfulness to the context
        3. Context quality and completeness
        
        lease grade the following response based on:
        1. Relevancy (0-1): How well does it answer the question?
        2. Faithfulness (0-1): How well does it stick to the provided context?
        3. Context Quality (0-1): How complete and relevant is the provided context?
        
        Question: {ctx.deps.query}
        Context: {ctx.deps.context}
        Response: {ctx.deps.response}
        
        Also determine if web search is needed to augment the context.
        
        Provide the grades and explanation in the JSON format with key atrributes 'Relevancy','Faithfulness','Context Quality','Needs Web Search':
        {"Relevancy": <score>,
        "Faithfulness": <score>,
        "Context Quality": <score>,
        "Needs Web Search": <true/false>,
        "Explanation": <explanation>,
        "Answer":<provide response based on the context from the `retrievre_tool' if 'Need Web Search' value is 'false' otherwise Use the `websearch_tool` function to generate the final reaponse}"""
        ),
        )

@groq_agent.tool_plain
async def websearch_tool(question) -> str:  
    """check if the square is a winner"""
    # Step 1. Instantiating your TavilyClient
    tavily_client = TavilyClient()

    # Step 2. Executing a Q&A search query
    answer = tavily_client.qna_search(query=question)

    # Step 3. That's it! Your question has been answered!
    print(f"WEB SEARCH:{answer}")
    return answer

@groq_agent.tool
async def rertiever_tool( ctx: RunContext[Deps],question:str)-> List[str]:
  load_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding,collection_name="fibromyalgia")
  docs = load_vectorstore.similarity_search(question,k=3)
  documnets = [d.page_content for d in docs]
  print(f"RAG Retrieval:{documnets}")
  return documnets


query = "What is Fibromyalgia?"
response = groq_agent.run_sync(query)

