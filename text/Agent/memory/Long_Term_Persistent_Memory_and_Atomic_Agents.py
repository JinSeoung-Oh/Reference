### From https://generativeai.pub/build-smarter-ai-agents-with-long-term-persistent-memory-and-atomic-agents-415b1d2b23ff

"""
git clone https://github.com/KennyVaneetvelde/persistent-memory-agent-example
cd persistent-memory-agent-example

pipx install poetry
poetry install

[tool.poetry.dependencies]
python = "^3.10"
atomic-agents = "^1.0.15"
rich = "^13.9.4"
instructor = "^1.6.4"
openai = "^1.54.4"
pydantic = "^2.9.2"
chromadb = "^0.5.18"
numpy = "^2.1.3"
"""

OPENAI_API_KEY=your_api_key_here

from typing import Literal
from pydantic import Field
from datetime import datetime, timezone
from atomic_agents.lib.base.base_io_schema import BaseIOSchema


class BaseMemory(BaseIOSchema):
    """Base class for all memory types"""
    content: str = Field(..., description="Content of the memory")
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO format timestamp of when the memory was created",
    )

class CoreBioMemory(BaseMemory):
    """Core biographical information about the user"""
    memory_type: Literal["core_bio"] = Field(default="core_bio")

class EventMemory(BaseMemory):
    """Information about significant events or experiences"""
    memory_type: Literal["event"] = Field(default="event")

class WorkProjectMemory(BaseMemory):
    """Information about work projects and tasks"""
    memory_type: Literal["work_project"] = Field(default="work_project")

------------------------------------------------------------------------------------------------------------------
from pydantic import Field

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from chat_with_memory.services.chroma_db import ChromaDBService
from chat_with_memory.tools.memory_models import (
    BaseMemory,
    CoreBioMemory,
    EventMemory,
    WorkProjectMemory,
)


class MemoryStoreInputSchema(BaseIOSchema):
    """Schema for storing memories"""

    memory: BaseMemory = Field(..., description="Memory to store")


class MemoryStoreOutputSchema(BaseIOSchema):
    """Schema for memory storage output"""

    memory: BaseMemory = Field(..., description="Stored memory with generated ID")


class MemoryStoreConfig(BaseToolConfig):
    """Configuration for the MemoryStoreTool"""

    collection_name: str = Field(
        default="chat_memories", description="Name of the ChromaDB collection to use"
    )
    persist_directory: str = Field(
        default="./chroma_db", description="Directory to persist ChromaDB data"
    )


class MemoryStoreTool(BaseTool):
    """Tool for storing chat memories using ChromaDB"""

    input_schema = MemoryStoreInputSchema
    output_schema = MemoryStoreOutputSchema

    def __init__(self, config: MemoryStoreConfig = MemoryStoreConfig()):
        super().__init__(config)
        self.db_service = ChromaDBService(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
        )

    def run(self, params: MemoryStoreInputSchema) -> MemoryStoreOutputSchema:
        """Store a new memory in ChromaDB"""
        memory = params.memory

        # Map memory types to their storage representation
        memory_type_mapping = {
            CoreBioMemory: "core_memory",
            EventMemory: "event_memory",
            WorkProjectMemory: "work_project_memory",
        }

        # Get the specific memory type
        memory_type = memory_type_mapping.get(type(memory), "base_memory")

        # Base metadata with all values as strings
        metadata = {
            "timestamp": memory.timestamp,
            "memory_type": memory_type,
        }

        self.db_service.add_documents(
            documents=[memory.content], metadatas=[metadata]
        )

        return MemoryStoreOutputSchema(memory=memory.model_copy())

------------------------------------------------------------------------------------------------------------------
from typing import List, Optional, Literal, Union
from pydantic import Field
from datetime import datetime
import json

from atomic_agents.lib.base.base_tool import BaseTool, BaseToolConfig
from atomic_agents.lib.base.base_io_schema import BaseIOSchema
from chat_with_memory.services.chroma_db import ChromaDBService, QueryResult
from chat_with_memory.tools.memory_models import (
    CoreBioMemory,
    EventMemory,
    WorkProjectMemory,
    BaseMemory,
)


class MemoryQueryInputSchema(BaseIOSchema):
    """Schema for querying memories"""

    query: str = Field(..., description="Query string to find relevant memories")
    n_results: Optional[int] = Field(
        default=2, description="Number of similar memories to retrieve"
    )
    memory_type: Optional[str] = Field(
        default=None, description="Optional memory type to filter memories"
    )


class MemoryQueryOutputSchema(BaseIOSchema):
    """Schema for memory query output"""

    memories: List[BaseMemory] = Field(
        default_factory=list, description="Retrieved memories"
    )


class MemoryQueryConfig(BaseToolConfig):
    """Configuration for the MemoryQueryTool"""

    collection_name: str = Field(
        default="chat_memories", description="Name of the ChromaDB collection to use"
    )
    persist_directory: str = Field(
        default="./chroma_db", description="Directory to persist ChromaDB data"
    )


class MemoryQueryTool(BaseTool):
    """Tool for querying chat memories using ChromaDB"""

    input_schema = MemoryQueryInputSchema
    output_schema = MemoryQueryOutputSchema

    def __init__(self, config: MemoryQueryConfig = MemoryQueryConfig()):
        super().__init__(config)
        self.db_service = ChromaDBService(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
        )

    def run(self, params: MemoryQueryInputSchema) -> MemoryQueryOutputSchema:
        """Query for relevant memories using semantic search"""
        where_filter = None
        if params.memory_type:
            # Map query types to stored types
            type_mapping = {
                "core": "core_memory",
                "event": "event_memory",
                "work_project": "work_project_memory",
            }
            memory_type = type_mapping[params.memory_type]
            where_filter = {"memory_type": memory_type}

        try:
            results: QueryResult = self.db_service.query(
                query_text=params.query,
                n_results=params.n_results,
                where=where_filter,
            )

            # Map stored types back to memory classes
            memory_class_mapping = {
                "core_memory": CoreBioMemory,
                "event_memory": EventMemory,
                "work_project_memory": WorkProjectMemory,
                "base_memory": BaseMemory,
            }

            memories = []
            if results["documents"]:
                for doc, meta, id_ in zip(
                    results["documents"], results["metadatas"], results["ids"]
                ):
                    memory_type = meta.get("memory_type", "base_memory")
                    memory_class = memory_class_mapping[memory_type]

                    base_data = {
                        "id": id_,
                        "content": doc,
                        "timestamp": meta["timestamp"],
                    }
                    memories.append(memory_class(**base_data))

            return MemoryQueryOutputSchema(memories=memories)
        except Exception as e:
            print(f"Query error: {str(e)}")
            return MemoryQueryOutputSchema(memories=[])
          
------------------------------------------------------------------------------------------------------------------
### Context Providers
from atomic_agents.lib.components.system_prompt_generator import SystemPromptContextProviderBase

class MemoryContextProvider(SystemPromptContextProviderBase):
    """Provides relevant memories as context for the agent"""
    def __init__(self, memories: List[BaseMemory]):
        super().__init__(title="Relevant Memories")
        self.memories = memories
    def get_info(self) -> str:
        if not self.memories:
            return "No relevant memories found."
        
        memory_strings = []
        for memory in self.memories:
            memory_strings.append(f"- {memory.content} ({memory.memory_type})")
        
        return "Previous memories:\n" + "\n".join(memory_strings)

------------------------------------------------------------------------------------------------------------------
############ Memory Formation Agent
import instructor
from openai import OpenAI
import os
from typing import List, Literal, Optional, Union, Dict, Any
from pydantic import Field
from datetime import datetime, timezone

from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory
from chat_with_memory.tools.memory_models import (
    BaseMemory,
    CoreBioMemory,
    EventMemory,
    WorkProjectMemory,
)
from chat_with_memory.tools.memory_store_tool import (
    MemoryStoreTool,
    MemoryStoreInputSchema,
)
from chat_with_memory.tools.memory_query_tool import (
    MemoryQueryTool,
    MemoryQueryInputSchema,
)


class MemoryFormationInputSchema(BaseIOSchema):
    """Input schema for the Memory Formation Agent."""

    last_user_msg: str = Field(
        ...,
        description="The last message from the user in the conversation",
    )
    last_assistant_msg: str = Field(
        ...,
        description="The last message from the assistant in the conversation",
    )


class MemoryFormationOutputSchema(BaseIOSchema):
    """Output schema for the Memory Formation Agent, representing the assistant's memory about the user."""

    reasoning: List[str] = Field(
        ...,
        description="Reasoning about which memory type to pick from the list of possible memory types and why",
        min_length=3,
        max_length=5,
    )
    memories: Optional[List[CoreBioMemory | EventMemory | WorkProjectMemory]] = Field(
        ...,
        description="The formed memories of the assistant about the user, if anything relevant was found.",
    )


# Initialize the system prompt generator with more selective criteria
memory_formation_prompt = SystemPromptGenerator(
    background=[
        "You are an AI specialized in identifying and preserving truly significant, long-term relevant information about users.",
        "You focus on extracting information that will remain relevant and useful over extended periods.",
        "You carefully filter out temporary states, trivial events, and time-bound information.",
        "You carefully filter out any memories that are already in the memory store.",
        "You understand the difference between temporarily relevant details and permanently useful knowledge.",
    ],
    steps=[
        "Analyze both the user's message and the assistant's message for context",
        "Consider the conversation flow to better understand the information's significance",
        "Look for information meeting these criteria:",
        "  - Permanent or long-lasting relevance (e.g., traits, background, significant relationships)",
        "  - Important biographical details (e.g., health conditions, cultural background)",
        "  - Major life events that shape the user's context",
        "  - Information that would be valuable months or years from now",
        "Filter out information that is:",
        "  - Temporary or time-bound",
        "  - Trivial daily events",
        "  - Current activities or states",
        "  - Administrative or routine matters",
        "  - Already in the existing memories",
        "For each truly significant piece of information:",
        "  - Formulate it in a way that preserves long-term relevance",
        "  - Choose the appropriate memory type",
        "  - Express it clearly and timelessly",
    ],
    output_instructions=[
        "Create memories only for information with lasting significance",
        "Do not create memories of things that are already in the memory store",
        "Format memories to be relevant regardless of when they are accessed",
        "Focus on permanent traits, important relationships, and significant events",
        "Exclude temporary states and trivial occurrences",
        "When in doubt, only store information that would be valuable in future conversations",
    ],
)

# Create the agent configuration
memory_formation_config = BaseAgentConfig(
    client=instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))),
    model="gpt-4o-mini",
    memory=AgentMemory(max_messages=10),
    system_prompt_generator=memory_formation_prompt,
    input_schema=MemoryFormationInputSchema,
    output_schema=MemoryFormationOutputSchema,
)

# Create the memory formation agent
memory_formation_agent = BaseAgent(memory_formation_config)

------------------------------------------------------------------------------------------------------------------
## Chat Agent
from atomic_agents.agents.base_agent import BaseAgent, BaseAgentConfig, BaseIOSchema
from atomic_agents.lib.components.system_prompt_generator import SystemPromptGenerator
from atomic_agents.lib.components.agent_memory import AgentMemory
from pydantic import Field

class ChatAgentInputSchema(BaseIOSchema):
    """Input schema for the Chat Agent."""
    message: str = Field(..., description="The user's message")
class ChatAgentOutputSchema(BaseIOSchema):
    """Output schema for the Chat Agent."""
    response: str = Field(..., description="The assistant's response to the user")
# Initialize the system prompt generator for the chat agent
chat_prompt = SystemPromptGenerator(
    background=[
        "You are a friendly and helpful AI assistant with access to long-term memories about the user.",
        "You use these memories to provide personalized and contextually relevant responses.",
        "You maintain a natural, conversational tone while being professional and respectful.",
    ],
    steps=[
        "Review any relevant memories about the user",
        "Consider the current context of the conversation",
        "Formulate a response that incorporates relevant memories naturally",
        "Ensure the response is helpful and moves the conversation forward",
    ],
    output_instructions=[
        "Keep responses concise but informative",
        "Reference memories naturally, as a human friend would",
        "Maintain a consistent personality across conversations",
        "Be helpful while respecting boundaries",
    ],
)
# Create the chat agent configuration
chat_agent_config = BaseAgentConfig(
    client=instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY"))),
    model="gpt-4o-mini",
    memory=AgentMemory(max_messages=10),
    system_prompt_generator=chat_prompt,
    input_schema=ChatAgentInputSchema,
    output_schema=ChatAgentOutputSchema,
)
# Create the chat agent
chat_agent = BaseAgent(chat_agent_config)

-------------------------------------------------------------------------------------------------
## main.py
def main() -> None:
    console = Console()
    store_tool = MemoryStoreTool()

    # Initialize tools and context providers
    memory_context_provider = MemoryContextProvider(
        title="Existing Memories",
    )
    current_date_context_provider = CurrentDateContextProvider(
        title="Current Date",
    )
    # Register context providers with agents
    chat_agent.register_context_provider("memory", memory_context_provider)
    chat_agent.register_context_provider("current_date", current_date_context_provider)
    memory_formation_agent.register_context_provider("memory", memory_context_provider)
    memory_formation_agent.register_context_provider(
        "current_date", current_date_context_provider
    )
    # Main conversation loop
    while True:
        # Get user input
        user_input = input("User: ")
        # Query relevant memories
        memory_query_tool = MemoryQueryTool()
        retrieved_memories = memory_query_tool.run(
            MemoryQueryInputSchema(query=user_input, n_results=10)
        )
        memory_context_provider.memories = retrieved_memories.memories
        # Form new memories if needed
        memory_assessment = memory_formation_agent.run(
            MemoryFormationInputSchema(
                last_user_msg=user_input,
                last_assistant_msg=last_assistant_msg
            )
        )
        # Store any new memories
        if memory_assessment.memories:
            for memory in memory_assessment.memories:
                store_tool.run(MemoryStoreInputSchema(memory=memory))
        # Generate chat response
        chat_response = chat_agent.run(ChatAgentInputSchema(message=user_input))
        last_assistant_msg = chat_response.response
        print(f"Assistant: {chat_response.response}")



