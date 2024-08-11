### From https://medium.com/aimonks/agentic-rag-with-llama-index-router-query-engine-01-381e83a418af

$ poetry init
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
$ poetry add python-dotenv ipykernel llama-index nest_asyncio

import dotenv
%load_ext dotenv
%dotenv

import nest_asyncio
nest_asyncio.apply()

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool

documents = SimpleDirectoryReader(input_files=["./datasets/lora_paper.pdf"]).load_data()
splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
node_metadata = nodes[1].get_content(metadata_mode=True)

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)

# summary query engine
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)

# vector query engine
vector_query_engine = vector_index.as_query_engine()

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to the Lora paper."
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the the Lora paper."
    ),
)

############### Router Query Engine ################
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector


query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)
response = query_engine.query("What is the summary of the document?")
print(str(response))


###### Full #######
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader

async def create_router_query_engine(
    document_fp: str,
    verbose: bool = True,
) -> RouterQueryEngine:
    # load lora_paper.pdf documents
    documents = SimpleDirectoryReader(input_files=[document_fp]).load_data()
    
    # chunk_size of 1024 is a good default value
    splitter = SentenceSplitter(chunk_size=1024)
    # Create nodes from documents
    nodes = splitter.get_nodes_from_documents(documents)
    
    # LLM model
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    # embedding model
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    
    # summary index
    summary_index = SummaryIndex(nodes)
    # vector store index
    vector_index = VectorStoreIndex(nodes)
    
    # summary query engine
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )

    # vector query engine
    vector_query_engine = vector_index.as_query_engine()
    
    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Useful for summarization questions related to the Lora paper."
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Useful for retrieving specific context from the the Lora paper."
        ),
    )
    
    
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
        ],
        verbose=verbose
    )
    
    
    return query_engine

query_engine = await create_router_query_engine("./datasets/lora_paper.pdf")
response = query_engine.query("What is the summary of the document?")
print(str(response))





