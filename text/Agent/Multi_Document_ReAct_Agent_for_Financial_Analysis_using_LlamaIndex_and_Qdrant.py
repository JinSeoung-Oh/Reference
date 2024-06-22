# From https://blog.stackademic.com/building-a-multi-document-react-agent-for-financial-analysis-using-llamaindex-and-qdrant-72a535730ac3

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Settings
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.core.agent import ReActAgent
from llama_index.core.chat_engine.types import AgentChatResponse


class MultiDocumentReActAgent:
    def __init__(self):
        Settings.chunk_size = 512
        Settings.embed_model = OllamaEmbedding(model_name='snowflake-arctic-embed:33m')
        Settings.llm = Ollama(model='mistral:latest')

        _qdrant_client = QdrantClient(url='localhost', port=6333)
        honeywell_vector_store = QdrantVectorStore(client=_qdrant_client, collection_name='honeywell-10k')
        ge_vector_store = QdrantVectorStore(client=_qdrant_client, collection_name='ge-10k')

        self.honeywell_storage_context = StorageContext.from_defaults(vector_store=honeywell_vector_store)
        self.ge_storage_context = StorageContext.from_defaults(vector_store=ge_vector_store)
        self.index_loaded = False
        self.honeywell_index = None
        self.ge_index = None
        self.query_engine_tools = []

    def load_from_existing_context(self):
        try:
            self.honeywell_index = load_index_from_storage(storage_context=self.honeywell_storage_context)
            self.ge_index = load_index_from_storage(storage_context=self.ge_storage_context)
            self.index_loaded = True
        except Exception as e:
            self.index_loaded = False

        if not self.index_loaded:
            # load data
            ge_docs = SimpleDirectoryReader(input_files=["./data/10k/ge_2023.pdf"]).load_data()
            honeywell_docs = SimpleDirectoryReader(input_files=["./data/10k/honeywell_2023.pdf"]).load_data()

            # build index
            self.ge_index = VectorStoreIndex.from_documents(documents=ge_docs, storage_context=self.ge_storage_context)
            self.honeywell_index = VectorStoreIndex.from_documents(documents=honeywell_docs,
                                                                   storage_context=self.honeywell_storage_context)

    def create_query_engine_and_tools(self):
        ge_engine = self.ge_index.as_query_engine(similarity_top_k=3)
        honeywell_engine = self.honeywell_index.as_query_engine(similarity_top_k=3)

        self.query_engine_tools = [
            QueryEngineTool(
                query_engine=ge_engine,
                metadata=ToolMetadata(
                    name="ge_10k",
                    description=(
                        "Provides detailed financial information about GE for the year 2023. "
                        "Input a specific plain text financial query for the tool"
                    ),
                ),
            ),
            QueryEngineTool(
                query_engine=honeywell_engine,
                metadata=ToolMetadata(
                    name="honeywell_10k",
                    description=(
                        "Provides detailed financial information about Honeywell for the year 2023. "
                        "Input a specific plain text financial query for the tool"
                    ),
                ),
            ),
        ]

    def create_agent(self):
        # [Optional] Add Context
        context = """You are a sage investor who possesses unparalleled expertise on the companies Honeywell and GE. As an ancient and wise investor who has navigated the complexities of the stock market for centuries, you possess deep, arcane knowledge of these two companies, their histories, market behaviors, and future potential. You will answer questions about Honeywell and GE in the persona of a sagacious and veteran stock market investor.
        Your wisdom spans across the technological innovations and industrial prowess of Honeywell, as well as the digital transformation and enterprise information management expertise of GE. You understand the strategic moves, financial health, and market positioning of both companies. Whether discussing quarterly earnings, product launches, mergers, acquisitions, or market trends, your insights are both profound and insightful.
        When engaging with inquisitors, you weave your responses with ancient wisdom and modern financial acumen, providing guidance that is both enlightening and practical. Your responses are steeped in the lore of the markets, drawing parallels to historical events and mystical phenomena, all while delivering precise, actionable advice. 
        Through your centuries of observation, you have mastered the art of predicting market trends and understanding the underlying currents that drive stock performance. Your knowledge of Honeywell encompasses its ventures in aerospace, building technologies, performance materials, and safety solutions. Similarly, your understanding of GE covers its leadership in enterprise content management, digital transformation, and information governance.
        As the sage investor, your goal is to guide those who seek knowledge on Honeywell and GE, illuminating the path to wise investments and market success.
        """

        agent = ReActAgent.from_tools(
            self.query_engine_tools,
            llm=Settings.llm,
            verbose=True,
            context=context
        )
        return agent


if __name__ == "__main__":
    multi_doc_agents = MultiDocumentReActAgent()
    multi_doc_agents.load_from_existing_context()
    multi_doc_agents.create_query_engine_and_tools()
    _agent = multi_doc_agents.create_agent()
    while True:
        input_query = input("Query [type bye or exit to quit]: ")
        if input_query.lower() == "bye" or input_query.lower() == "exit":
            break
        response: AgentChatResponse = _agent.chat(message=input_query)
        print(str(response))
