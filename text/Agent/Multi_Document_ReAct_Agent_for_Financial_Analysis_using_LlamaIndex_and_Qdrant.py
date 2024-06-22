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
