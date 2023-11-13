#from https://medium.com/@sauravjoshi23/complex-query-resolution-through-llamaindex-utilizing-recursive-retrieval-document-agents-and-sub-d4861ecd54e6

## LlamaIndex's system for deal with complex query : 
#Recursive Retrieval with Document Agents, and Sub Question Query Decomposition to adeptly navigate through, 
#and make decisions over, heterogeneous documents. 
#This system not only retrieves pertinent information but also breaks down complex queries into manageable sub-questions, 
#delegates them to specialized document agents equipped with query engine tools such as vector query engine, 
#summary query engine, and knowledge graph query engine, and synthesizes the retrieved information into a coherent, comprehensive response

## Recursive Retriever
#Recursive Retriever instead of finding and using small, isolated pieces of information/text chunks to answer queries, 
#it aims to use summaries of entire documents, providing answers 
#that are not only accurate but also maintain a better understanding and representation of the overall context or topic being discussed or queried

## Document Agents
#Document Agents are designed to dynamically perform tasks beyond mere fact-based question-answering within a document. 

## Sub Question Query Engine
#The Sub Question Query Engine aims at deconstructing a complex query into more manageable sub-questions, 
#each tailored to extract specific information from relevant data sources.

## Query Engine Tools
#Query engine is a generic interface that allows you to ask question over your data. A query engine takes in a natural language query, 
#and returns a rich response. Three distinct query engines are constructed to manage different aspects of information retrieval.
#1. vector_query_engine
#   Derived from a VectorStoreIndex, focusing on efficiently retrieving relevant document sections based on vector similarity
#2. list_query_engine
#   Sourced from a SummaryIndex, emphasizes fetching summarized information, ensuring concise and relevant data extraction
#3. graph_query_engine
#   Originating from a KnowledgeGraphIndex, is efficient at extracting structured, interconnected, and relational knowledge

## Implementation
%pip install llama-index pinecone-client transformers neo4j python-dotenv
import os
import torch
import pinecone
from transformers import pipeline
from llama_index import (
    VectorStoreIndex,
    SummaryIndex,
    KnowledgeGraphIndex,
    SimpleKeywordTableIndex,
    SimpleDirectoryReader,
    ServiceContext,
    StorageContext
)
from dotenv import load_dotenv
from llama_index.schema import IndexNode
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.llms import OpenAI
from llama_index.query_engine import SubQuestionQueryEngine
from llama_index.retrievers import RecursiveRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response_synthesizers import get_response_synthesizer
from llama_index.vector_stores import PineconeVectorStore
from llama_index.graph_stores import Neo4jGraphStore

#DataPreparation
wiki_titles = ["Seattle", "Boston", "Chicago"]

from pathlib import Path

import requests

for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            # 'exintro': True,
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    data_path = Path("data")
    if not data_path.exists():
        Path.mkdir(data_path)

    with open(data_path / f"{title}.txt", "w") as fp:
        fp.write(wiki_text)

# Load all wiki documents
city_docs = {}
for wiki_title in wiki_titles:
    city_docs[wiki_title] = SimpleDirectoryReader(
        input_files=[f"data/{wiki_title}.txt"]
    ).load_data()

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm)

# Build Document Agent for each Document
# Vector Storage Context
# init pinecone
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')
os.environ["PINECONE_ENVIRONMENT"] = os.getenv('PINECONE_ENVIRONMENT')
pinecone.init(api_key=os.environ["PINECONE_API_KEY"], environment=os.environ["PINECONE_ENVIRONMENT"])
pinecone.create_index("vector-index", dimension=1536, metric="euclidean", pod_type="p1")

# construct vector store and customize storage context
vector_storage_context = StorageContext.from_defaults(
    vector_store=PineconeVectorStore(pinecone.Index("vector-index"))
)

#Graph Storage Context
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

triplet_extractor = pipeline(
    'text2text-generation',
    model='Babelscape/rebel-large',
    tokenizer='Babelscape/rebel-large',
    device=device)

import re

def clean_triplets(input_text, triplets):
    """Sometimes the model hallucinates, so we filter out entities
       not present in the text"""
    text = input_text.lower()
    clean_triplets = []
    for triplet in triplets:

        if (triplet["head"] == triplet["tail"]):
            continue

        head_match = re.search(
            r'\b' + re.escape(triplet["head"].lower()) + r'\b', text)
        if head_match:
            head_index = head_match.start()
        else:
            head_index = text.find(triplet["head"].lower())

        tail_match = re.search(
            r'\b' + re.escape(triplet["tail"].lower()) + r'\b', text)
        if tail_match:
            tail_index = tail_match.start()
        else:
            tail_index = text.find(triplet["tail"].lower())

        if ((head_index == -1) or (tail_index == -1)):
            continue

        clean_triplets.append((triplet["head"], triplet["type"], triplet["tail"]))

    return clean_triplets

def extract_triplets(input_text):
    text = triplet_extractor.tokenizer.batch_decode([triplet_extractor(input_text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])[0]

    triplets = []
    relation, subject, relation, object_ = '', '', '', ''
    text = text.strip()
    current = 'x'
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token

    if subject != '' and relation != '' and object_ != '':
        triplets.append({'head': subject.strip(), 'type': relation.strip(), 'tail':object_.strip()})
    clean = clean_triplets(input_text, triplets)
    return clean

os.environ["NEO4J_URI"] = os.getenv('NEO4J_URI')
os.environ["NEO4J_USERNAME"] = os.getenv('NEO4J_USERNAME')
os.environ["NEO4J_PASSWORD"] = os.getenv('NEO4J_PASSWORD')
os.environ["NEO4J_DB"] = os.getenv('NEO4J_DB')

graph_store = Neo4jGraphStore(
    username=os.environ["NEO4J_USERNAME"],
    password=os.environ["NEO4J_PASSWORD"],
    url=os.environ["NEO4J_URI"],
    database=os.environ["NEO4J_DB"],
)

graph_storage_context = StorageContext.from_defaults(graph_store=graph_store)

from llama_index.agent import OpenAIAgent

# Build agents dictionary
agents = {}

for wiki_title in wiki_titles:
    # build vector index
    vector_index = VectorStoreIndex.from_documents(
        city_docs[wiki_title], service_context=service_context, storage_context=vector_storage_context
    )
    # build summary index
    summary_index = SummaryIndex.from_documents(
        city_docs[wiki_title], service_context=service_context
    )
    # build graph index
    graph_index = KnowledgeGraphIndex.from_documents(
        city_docs[wiki_title],
        storage_context=graph_storage_context,
        kg_triplet_extract_fn=extract_triplets,
        service_context=ServiceContext.from_defaults(llm=llm, chunk_size=256)
    )
    # define query engines
    vector_query_engine = vector_index.as_query_engine()
    list_query_engine = summary_index.as_query_engine()
    graph_query_engine = graph_index.as_query_engine()

    # define tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=f"Useful for retrieving specific context from {wiki_title}",
            ),
        ),
        QueryEngineTool(
            query_engine=list_query_engine,
            metadata=ToolMetadata(
                name="summary_tool",
                description=f"Useful for summarization questions related to {wiki_title}",
            ),
        ),
        QueryEngineTool(
            query_engine=graph_query_engine,
            metadata=ToolMetadata(
                name="graph_tool",
                description=f"Useful for retrieving structural, interconnected and relational knowledge related to {wiki_title}",
            ),
        ),
    ]

    # build agent
    function_llm = OpenAI(model="gpt-3.5-turbo-0613")
    agent = OpenAIAgent.from_tools(
        query_engine_tools,
        llm=function_llm,
        verbose=True,
    )

    agents[wiki_title] = agent

# Build Recursive Retriever over these Agents
# define top-level nodes
nodes = []
for wiki_title in wiki_titles:
    wiki_summary = (
        f"This content contains Wikipedia articles about {wiki_title}. "
        f"Use this index if you need to lookup specific facts about {wiki_title}.\n"
        "Do not use this index if you want to analyze multiple cities."
    )
    node = IndexNode(text=wiki_summary, index_id=wiki_title)
    nodes.append(node)

# define top-level retriever
top_vector_index = VectorStoreIndex(nodes)
vector_retriever = top_vector_index.as_retriever(similarity_top_k=1)

# define recursive retriever
recursive_retriever = RecursiveRetriever(
    "vector",
    retriever_dict={"vector": vector_retriever},
    query_engine_dict=agents,
    verbose=True,
)

response_synthesizer = get_response_synthesizer(
    response_mode="compact",
)
retriever_query_engine = RetrieverQueryEngine.from_args(
    recursive_retriever,
    response_synthesizer=response_synthesizer,
    service_context=service_context,
)

# Setup sub question query engine
# convert the recursive retriever into a tool
query_engine_tools = [
    QueryEngineTool(
        query_engine=retriever_query_engine,
        metadata=ToolMetadata(
            name="recursive_retriever",
            description="Recursive retriever for accessing documents"
        ),
    ),
]

# setup sub question query engine
query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
)

response = query_engine.query(
    "Tell me about the sports teams in Boston and the positive aspects of Seattle"
)
print(response)

