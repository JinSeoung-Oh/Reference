## From https://medium.com/@amanatulla1606/introducing-transformers-agents-2-0-14a5601ade0b

"""
1. The Transformers Agents approach
   Building agent workflows is complex, and we feel these systems need a lot of clarity and modularity.
   HF launched Transformers Agents one year ago, and they doubling down on core design goals.

   * Framework strives for:
     -1. Clarity through simplicity
         reduce abstractions to the minimum. Simple error logs and accessible attributes let you easily inspect what’s happening and give you more clarity.
     -2. Modularity
         prefer to propose building blocks rather than full, complex feature sets. You are free to choose whatever building blocks are best for your project.
     -3. For instance, since any agent system is just a vehicle powered by an LLM engine, 
         they decided to conceptually separate the two, which lets you create any agent type from any underlying LLM.

2. Main elements
   -1. Tool
       this is the class that lets you use a tool or implement a new one. It is composed mainly of a callable forward method that executes the tool action, 
       and a set of a few essential attributes: name, descriptions, inputs and output_type. 
       These attributes are used to dynamically generate a usage manual for the tool and insert it into the LLM’s prompt.
   -2. Toolbox
       It's a set of tools that are provided to an agent as resources to solve a particular task. For performance reasons, 
       tools in a toolbox are already instantiated and ready to go. This is because some tools take time to initialize, 
       so it’s usually better to re-use an existing toolbox and just swap one tool, rather than re-building a set of tools from scratch at each agent initialization.
   -3. CodeAgent
       a very simple agent that generates its actions as one single blob of Python code. It will not be able to iterate on previous observations.
   -4. ReactAgent
       ReAct agents follow a cycle of Thought ⇒ Action ⇒ Observation until they’ve solve the task. We propose two classes of ReactAgent:
   -5. ReactCodeAgent generates its actions as python blobs.
   -6. ReactJsonAgent generates its actions as JSON blobs. 

"""

!pip install "git+https://github.com/huggingface/transformers.git#egg=transformers[agents]"
!pip install langchain sentence-transformers faiss-cpu langchain langchain_community datasets

import datasets
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import json
from transformers.agents import Tool
from langchain_core.vectorstores import VectorStore
from transformers.agents import HfEngine, ReactJsonAgent
from transformers.agents import ReactCodeAgent

knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

source_docs = [
    Document(
        page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]}
    ) for doc in knowledge_base
]

docs_processed = RecursiveCharacterTextSplitter(chunk_size=500).split_documents(source_docs)[:1000]

embedding_model = HuggingFaceEmbeddings(model_name="thenlper/gte-small")
vectordb = FAISS.from_documents(
    documents=docs_processed,
    embedding=embedding_model
)

all_sources = list(set([doc.metadata["source"] for doc in docs_processed]))

class RetrieverTool(Tool):
    name = "retriever"
    description = "Retrieves some documents from the knowledge base that have the closest embeddings to the input query."
    inputs = {
        "query": {
            "type": "text",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "source": {
            "type": "text", 
            "description": ""
        },
    }
    output_type = "text"
    
    def __init__(self, vectordb: VectorStore, all_sources: str, **kwargs):
        super().__init__(**kwargs)
        self.vectordb = vectordb
        self.inputs["source"]["description"] = (
            f"The source of the documents to search, as a str representation of a list. Possible values in the list are: {all_sources}. If this argument is not provided, all sources will be searched."
          )

    def forward(self, query: str, source: str = None) -> str:
        assert isinstance(query, str), "Your search query must be a string"

        if source:
            if isinstance(source, str) and "[" not in str(source): # if the source is not representing a list
                source = [source]
            source = json.loads(str(source).replace("'", '"'))

        docs = self.vectordb.similarity_search(query, filter=({"source": source} if source else None), k=3)

        if len(docs) == 0:
            return "No documents found with this filtering. Try removing the source filter."
        return "Retrieved documents:\n\n" + "\n===Document===\n".join(
            [doc.page_content for doc in docs]
        )

class SearchTool(Tool):
    name = "ask_search_agent"
    description = "A search agent that will browse the internet to answer a question. Use it to gather informations, not for problem-solving."

    inputs = {
        "question": {
            "description": "Your question, as a natural language sentence. You are talking to an agent, so provide them with as much context as possible.",
            "type": "text",
        }
    }
    output_type = "text"

    def forward(self, question: str) -> str:
        return websurfer_agent.run(question)

llm_engine = HfEngine("meta-llama/Meta-Llama-3-70B-Instruct")

agent = ReactJsonAgent(
    tools=[RetrieverTool(vectordb, all_sources)],
    llm_engine=llm_engine
)

agent_output = agent.run("Please show me a LORA finetuning script")

WEB_TOOLS = [
    SearchInformationTool(),
    NavigationalSearchTool(),
    VisitTool(),
    DownloadTool(),
    PageUpTool(),
    PageDownTool(),
    FinderTool(),
    FindNextTool(),
]

websurfer_llm_engine = HfEngine(
    model="CohereForAI/c4ai-command-r-plus"
)  # We choose Command-R+ for its high context length

websurfer_agent = ReactJsonAgent(
    tools=WEB_TOOLS,
    llm_engine=websurfer_llm_engine,
)

llm_engine = HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
react_agent_hf = ReactCodeAgent(
    tools=[SearchTool()],
    llm_engine=llm_engine,
)


