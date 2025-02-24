### From https://medium.com/ai-agent-insider/build-ai-agents-with-active-memory-management-using-langmem-6bdc38449f74

!pip install langmem

from langmem import Memory

# Initialize memory storage
memory = Memory(storage="faiss", namespace="user_123")

# Store a memory
memory.add("User likes AI-generated images and machine learning content.")

# Retrieve relevant memories
print(memory.retrieve("What does the user prefer?"))

# Create different namespaces for different users or teams
user_memory = Memory(storage="faiss", namespace="user_123")
team_memory = Memory(storage="faiss", namespace="team_codeb")

# Add memory to a team namespace
team_memory.add("Team CodeB.ai focuses on AI for renewable energy and waste management.")

# Agent A stores memory
agent_a = Memory(storage="faiss", namespace="shared_knowledge")
agent_a.add("LangMem helps AI agents retain long-term memory.")

# Agent B retrieves memory
agent_b = Memory(storage="faiss", namespace="shared_knowledge")
print(agent_b.retrieve("What is LangMem used for?"))

-----------------------------------------------------------------------------------------
from langchain.chat_models import ChatOpenAI
from langmem import Memory

# Initialize memory
memory = Memory(storage="faiss", namespace="user_456")

# Store user preferences
memory.add("User is interested in blockchain and smart contracts.")

# Initialize LLM with memory
llm = ChatOpenAI(temperature=0.7)
context = memory.retrieve("What topics does the user like?")

# Generate response with memory context
response = llm.predict(f"Given the user preferences: {context}, suggest an AI project idea.")
print(response)
