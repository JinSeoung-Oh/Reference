### From https://medium.com/@nomannayeem/building-ai-agents-that-actually-remember-a-developers-guide-to-memory-management-in-2025-062fd0be80a1
## Have to visit given link

-------------------------------------------------------------------------------------------------------------------------------------
########### Short-Term Memory
"""
Short-term memory is all about context within a single conversation. When a user says “I’m planning a trip to Japan,”
your AI should remember “Japan trip” for the rest of that chat session.
If they later ask “What’s the weather like there?”, your AI should know “there” means Japan.

Here’s what good short-term memory looks like in action:

User: “I’m looking for a restaurant in downtown Seattle.”
AI: “Great! What type of cuisine are you in the mood for in downtown Seattle?”
User: “Something Italian would be perfect.”
AI: “Perfect! I found some excellent Italian restaurants in downtown Seattle. Would you prefer fine dining or something more casual?”

Notice how the AI keeps track of:

Location: downtown Seattle
Food type: Italian
Context: Looking for restaurants
"""
#*****************************************************************************
## LangChain - Basic Conversation Memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# Create a memory system that remembers everything
memory = ConversationBufferMemory()

# Connect it to a conversation chain
conversation = ConversationChain(
    llm=OpenAI(temperature=0.7),
    memory=memory,
    verbose=True  # So you can see what's happening
)

# Now your AI remembers context!
response1 = conversation.predict(input="Hi, I'm Sarah and I love hiking")
print(response1)  # AI introduces itself and acknowledges Sarah likes hiking

response2 = conversation.predict(input="What outdoor activities would you recommend for me?")
print(response2)  # AI remembers Sarah likes hiking and suggests related activities

#*****************************************************************************
##  LangChain- Smart Memory with Summaries
from langchain.memory import ConversationSummaryMemory

# Create memory that summarizes old conversations
summary_memory = ConversationSummaryMemory(
    llm=OpenAI(temperature=0),
    max_token_limit=1000  # Start summarizing when we hit 1000 tokens
)

conversation_with_summary = ConversationChain(
    llm=OpenAI(temperature=0.7),
    memory=summary_memory,
    verbose=True
)

# The AI will automatically summarize older parts of the conversation
# while keeping recent messages in full detail

#*****************************************************************************
## LangChain- Sliding Window Memory (The Sweet Spot)
from langchain.memory import ConversationBufferWindowMemory

# Keep only the last 10 exchanges
window_memory = ConversationBufferWindowMemory(k=10)

conversation_with_window = ConversationChain(
    llm=OpenAI(temperature=0.7),
    memory=window_memory,
    verbose=True
)

# Now your AI remembers the last 10 messages perfectly
# and forgets everything older than that

#*****************************************************************************
## Pydantic AI - Structured Memory Management

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, SystemMessage, UserMessage
from typing import List

class ConversationManager:
    def __init__(self):
        self.agent = Agent(
            'openai:gpt-4',
            system_prompt='You are a helpful assistant with perfect memory of our conversation.'
        )
        self.conversation_history: List[ModelMessage] = []
    
    async def chat(self, user_input: str) -> str:
        # Add the user's message to our history
        user_message = UserMessage(content=user_input)
        self.conversation_history.append(user_message)
        
        # Get AI response with full conversation context
        result = await self.agent.run(
            user_input,
            message_history=self.conversation_history
        )
        
        # Add AI's response to history
        self.conversation_history.extend(result.new_messages())
        
        return result.data
    
    def get_conversation_summary(self) -> dict:
        """Get a summary of the current conversation"""
        return {
            "total_messages": len(self.conversation_history),
            "user_messages": len([msg for msg in self.conversation_history if isinstance(msg, UserMessage)]),
            "recent_topics": self._extract_recent_topics()
        }
    
    def _extract_recent_topics(self) -> List[str]:
        # You could implement topic extraction here
        return ["hiking", "outdoor activities"]  # Example

# Create a conversation manager
chat_manager = ConversationManager()

# Have a conversation with memory
response1 = await chat_manager.chat("Hi, I'm planning a camping trip")
print(response1)

response2 = await chat_manager.chat("What should I pack?")
print(response2)  # AI remembers you're planning a camping trip

# Check conversation stats
print(chat_manager.get_conversation_summary())

#*****************************************************************************
## Custom Memory Optimization
class OptimizedConversationManager(ConversationManager):
    def __init__(self, max_messages=20):
        super().__init__()
        self.max_messages = max_messages
    
    def _trim_history(self):
        """Keep conversation from getting too long"""
        if len(self.conversation_history) > self.max_messages:
            # Keep the first message (usually contains important context)
            # and the last max_messages-1 messages
            first_message = self.conversation_history[0]
            recent_messages = self.conversation_history[-(self.max_messages-1):]
            self.conversation_history = [first_message] + recent_messages
    
    async def chat(self, user_input: str) -> str:
        result = await super().chat(user_input)
        self._trim_history()  # Optimize after each exchange
        return result
""""
When to Use Each Type of Memory

Here’s a simple guide for choosing the right memory type:
-a. Buffer Memory (Remember Everything):
    - Short conversations (under 20 exchanges)
    - When you need perfect recall
    - Testing and development

-b. Summary Memory (Smart Compression):
    - Long conversations where context matters
    - Educational applications
    - Complex problem-solving sessions

-c. Window Memory (Last N Messages):
    - Customer service chats
    - General-purpose assistants
    - Most production applications

-d. Custom Memory (Pydantic AI Style):
    - When you need full control
    - Complex applications with specific memory needs
    - When you want to implement your own optimization
"""
-------------------------------------------------------------------------------------------------------------------------------------

########### Long-Term Memory
"""
Short-term memory is like a sticky note, but long-term memory is like a filing cabinet. 
While short-term memory helps your AI stay coherent during a single conversation, 
long-term memory is what transforms your bot from a helpful tool into something that feels almost human.

Think about it: when you talk to a good friend, they remember your birthday, your job situation, that funny story you told three months ago,
                and the fact that you hate cilantro. 
                That’s the kind of experience users want from AI — and long-term memory is how you deliver it.

"""
#*****************************************************************************
## LangChain’s Entity Memory: Remembering What Matters
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

# Create entity memory that tracks important information
entity_memory = ConversationEntityMemory(
    llm=OpenAI(temperature=0),
    k=10,  # Remember details about the 10 most recent entities
    entity_extraction_prompt=None,  # Use default entity extraction
    entity_summarization_prompt=None  # Use default summarization
)

# Set up the conversation with entity memory
conversation = ConversationChain(
    llm=OpenAI(temperature=0.7),
    memory=entity_memory,
    prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    verbose=True
)

# Now watch the magic happen
response1 = conversation.predict(
    input="Hi, I'm Sarah from TechCorp. I'm working on a project about sustainable energy with my colleague Mike."
)

response2 = conversation.predict(
    input="How's Mike doing with the sustainable energy research?"
)
# The AI remembers Sarah works at TechCorp, is collaborating with Mike, 
# and they're focused on sustainable energy

import json
from typing import Dict, Any

class PersistentEntityMemory:
    def __init__(self, storage_file: str = "entity_memory.json"):
        self.storage_file = storage_file
        self.entities = self.load_entities()
        
        # Set up LangChain entity memory
        self.langchain_memory = ConversationEntityMemory(
            llm=OpenAI(temperature=0),
            k=10
        )
        
        # Load existing entities into LangChain memory
        self.langchain_memory.entity_store = self.entities
    
    def load_entities(self) -> Dict[str, Any]:
        """Load entities from storage"""
        try:
            with open(self.storage_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_entities(self):
        """Save current entities to storage"""
        with open(self.storage_file, 'w') as f:
            json.dump(self.langchain_memory.entity_store, f, indent=2)
    
    def remember_user(self, user_id: str, conversation_text: str):
        """Process and remember information about a user"""
        # Extract entities from conversation
        entities = self.langchain_memory.llm.predict(
            f"Extract key information about people, companies, and preferences from: {conversation_text}"
        )
        
        # Store user-specific information
        if user_id not in self.entities:
            self.entities[user_id] = {}
        
        # Update entity information
        self.entities[user_id].update({
            "last_conversation": conversation_text[:200],  # Keep snippet
            "last_seen": self.get_current_timestamp(),
            "conversation_count": self.entities[user_id].get("conversation_count", 0) + 1
        })
        
        # Save to persistent storage
        self.save_entities()
    
    def get_user_context(self, user_id: str) -> str:
        """Get relevant context about a user"""
        if user_id not in self.entities:
            return "This appears to be a new user."
        
        user_info = self.entities[user_id]
        return f"""
        Previous context about this user:
        - Last seen: {user_info.get('last_seen', 'Unknown')}
        - Conversations: {user_info.get('conversation_count', 0)}
        - Last topic: {user_info.get('last_conversation', 'No previous context')}
        """

# Usage example
memory_system = PersistentEntityMemory()

# When user starts conversation
user_context = memory_system.get_user_context("sarah_123")
print(user_context)

# After conversation ends
memory_system.remember_user("sarah_123", "Sarah discussed her new role as marketing director at TechCorp")

#*****************************************************************************
## Pydantic AI with Vector Memory
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
from datetime import datetime

class MemoryEntry(BaseModel):
    user_id: str
    content: str
    timestamp: datetime
    conversation_id: str
    topics: List[str] = []
    importance_score: float = 0.5

class VectorMemorySystem:
    def __init__(self):
        self.agent = Agent('openai:gpt-4')
        self.memory_store: List[MemoryEntry] = []
        self.load_memory()
    
    def load_memory(self):
        """Load existing memories from storage"""
        try:
            with open('vector_memory.json', 'r') as f:
                data = json.load(f)
                self.memory_store = [MemoryEntry(**entry) for entry in data]
        except FileNotFoundError:
            self.memory_store = []
    
    def save_memory(self):
        """Save memories to persistent storage"""
        with open('vector_memory.json', 'w') as f:
            json.dump([entry.dict() for entry in self.memory_store], f, indent=2, default=str)
    
    async def store_memory(self, user_id: str, content: str, conversation_id: str):
        """Store a new memory with automatic topic extraction"""
        # Extract topics using AI
        topic_result = await self.agent.run(
            f"Extract 3-5 key topics or themes from this text. Return as comma-separated list: {content}"
        )
        topics = [topic.strip() for topic in topic_result.data.split(',')]
        
        # Calculate importance (you could make this more sophisticated)
        importance = await self._calculate_importance(content)
        
        # Create and store memory
        memory = MemoryEntry(
            user_id=user_id,
            content=content,
            timestamp=datetime.now(),
            conversation_id=conversation_id,
            topics=topics,
            importance_score=importance
        )
        
        self.memory_store.append(memory)
        self.save_memory()
    
    async def _calculate_importance(self, content: str) -> float:
        """Calculate how important this memory is (0.0 to 1.0)"""
        # Simple importance scoring - you could make this much more sophisticated
        importance_keywords = ['problem', 'issue', 'urgent', 'important', 'deadline', 'project']
        
        score = 0.5  # baseline
        content_lower = content.lower()
        
        for keyword in importance_keywords:
            if keyword in content_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    async def recall_memories(self, user_id: str, query: str, limit: int = 5) -> List[MemoryEntry]:
        """Find relevant memories for a user and query"""
        user_memories = [m for m in self.memory_store if m.user_id == user_id]
        
        if not user_memories:
            return []
        
        # Simple keyword matching (in production, you'd use vector similarity)
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in user_memories:
            # Calculate relevance score
            memory_words = set(memory.content.lower().split())
            topic_words = set(' '.join(memory.topics).lower().split())
            
            keyword_overlap = len(query_words.intersection(memory_words))
            topic_overlap = len(query_words.intersection(topic_words))
            
            relevance_score = (keyword_overlap * 0.7) + (topic_overlap * 0.3) + memory.importance_score
            
            if relevance_score > 0:
                scored_memories.append((relevance_score, memory))
        
        # Sort by relevance and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:limit]]
    
    async def chat_with_memory(self, user_id: str, message: str, conversation_id: str) -> str:
        """Chat with full memory context"""
        # Recall relevant memories
        relevant_memories = await self.recall_memories(user_id, message)
        
        # Build context from memories
        memory_context = ""
        if relevant_memories:
            memory_context = "Previous context:\n"
            for memory in relevant_memories:
                memory_context += f"- {memory.timestamp.strftime('%Y-%m-%d')}: {memory.content[:100]}...\n"
        
        # Generate response with context
        full_prompt = f"{memory_context}\nCurrent message: {message}"
        response = await self.agent.run(full_prompt)
        
        # Store this interaction
        await self.store_memory(
            user_id=user_id,
            content=f"User: {message}\nAssistant: {response.data}",
            conversation_id=conversation_id
        )
        
        return response.data

# Usage example
memory_system = VectorMemorySystem()

# First conversation
response1 = await memory_system.chat_with_memory(
    user_id="sarah_123",
    message="I'm working on a machine learning project for my company TechCorp",
    conversation_id="conv_001"
)

# Later conversation (different session)
response2 = await memory_system.chat_with_memory(
    user_id="sarah_123", 
    message="How's my ML project coming along?",
    conversation_id="conv_002"
)
# The system will remember Sarah works at TechCorp on an ML project!

#*****************************************************************************
## Agno: Production-Grade Memory Architecture
from agno import Agent, AgentMemory
from agno.memory import VectorMemory
import asyncio

class ProductionMemoryAgent:
    def __init__(self, user_id: str):
        self.user_id = user_id
        
        # Create user-specific vector memory
        self.memory = VectorMemory(
            collection_name=f"user_{user_id}",
            embeddings_model="text-embedding-3-large",
            distance_metric="cosine"
        )
        
        # Create agent with memory
        self.agent = Agent(
            model="gpt-4-turbo",
            memory=self.memory,
            instructions=f"""
            You are an AI assistant with perfect memory of your conversations with this user.
            Always reference relevant past conversations when appropriate.
            Be personal and build on previous interactions.
            """
        )
    
    async def chat(self, message: str) -> str:
        """Chat with full memory integration"""
        # Agno automatically handles memory retrieval and storage
        response = await self.agent.run(message)
        return response
    
    async def get_memory_summary(self) -> dict:
        """Get insights about stored memories"""
        memories = await self.memory.search(
            query="",  # Empty query to get all memories
            limit=100
        )
        
        return {
            "total_memories": len(memories),
            "conversation_topics": await self._extract_topics(memories),
            "first_interaction": memories[-1]["metadata"]["timestamp"] if memories else None,
            "last_interaction": memories[0]["metadata"]["timestamp"] if memories else None
        }
    
    async def _extract_topics(self, memories: list) -> list:
        """Extract common topics from memory"""
        if not memories:
            return []
        
        # Combine recent memory content
        recent_content = " ".join([mem["content"] for mem in memories[:20]])
        
        # Use agent to extract topics
        topic_response = await self.agent.run(
            f"Extract the top 5 topics discussed in these conversations: {recent_content}"
        )
        
        return topic_response.split('\n')[:5]

# Usage for production
async def main():
    # Create memory-enabled agents for different users
    sarah_agent = ProductionMemoryAgent("sarah_123")
    john_agent = ProductionMemoryAgent("john_456")
    
    # Sarah's conversation
    sarah_response = await sarah_agent.chat("I'm launching a new product at TechCorp next month")
    print(f"Sarah: {sarah_response}")
    
    # John's separate conversation  
    john_response = await john_agent.chat("I need help with my marketing strategy")
    print(f"John: {john_response}")
    
    # Later - Sarah returns
    sarah_response2 = await sarah_agent.chat("How should I prepare for the product launch?")
    print(f"Sarah (later): {sarah_response2}")  # Will reference TechCorp and product launch
    
    # Check Sarah's memory summary
    summary = await sarah_agent.get_memory_summary()
    print(f"Sarah's memory: {summary}")

# Run the example
asyncio.run(main())

-----------------------------------------------------------------------------------------------------------------
### Multi-Modal Memory: Beyond Just Text
from langchain.memory import (
    CombinedMemory,
    ConversationSummaryMemory,
    ConversationEntityMemory,
    ConversationKGMemory
)
from langchain.chains import ConversationChain
from langchain.llms import OpenAI

class AdvancedMemorySystem:
    def __init__(self):
        self.llm = OpenAI(temperature=0.7)
        
        # Different types of memory working together
        self.summary_memory = ConversationSummaryMemory(
            llm=self.llm,
            max_token_limit=1000
        )
        
        self.entity_memory = ConversationEntityMemory(
            llm=self.llm,
            k=15
        )
        
        # Knowledge graph memory - tracks relationships between concepts
        self.kg_memory = ConversationKGMemory(
            llm=self.llm,
            k=10
        )
        
        # Combine all memory types
        self.combined_memory = CombinedMemory(
            memories=[
                self.summary_memory,
                self.entity_memory, 
                self.kg_memory
            ]
        )
        
        # Track user patterns (custom addition)
        self.user_patterns = {}
    
    def track_interaction_pattern(self, user_id: str, interaction_data: dict):
        """Track patterns in how users interact"""
        if user_id not in self.user_patterns:
            self.user_patterns[user_id] = {
                'preferred_response_length': 'medium',
                'typical_session_duration': 0,
                'common_topics': [],
                'interaction_times': [],
                'complexity_preference': 'intermediate'
            }
        
        patterns = self.user_patterns[user_id]
        
        # Update patterns based on interaction
        if 'response_length_preference' in interaction_data:
            patterns['preferred_response_length'] = interaction_data['response_length_preference']
        
        if 'session_duration' in interaction_data:
            # Running average of session duration
            current_avg = patterns['typical_session_duration']
            new_duration = interaction_data['session_duration']
            patterns['typical_session_duration'] = (current_avg * 0.8) + (new_duration * 0.2)
        
        if 'topics' in interaction_data:
            # Track topic frequency
            for topic in interaction_data['topics']:
                if topic not in patterns['common_topics']:
                    patterns['common_topics'].append(topic)
    
    def get_personalized_context(self, user_id: str) -> str:
        """Generate personalized context based on user patterns"""
        if user_id not in self.user_patterns:
            return ""
        
        patterns = self.user_patterns[user_id]
        
        context = f"""
        User Interaction Preferences:
        - Prefers {patterns['preferred_response_length']} length responses
        - Typically engages for {patterns['typical_session_duration']:.1f} minutes
        - Common topics: {', '.join(patterns['common_topics'][:5])}
        - Complexity level: {patterns['complexity_preference']}
        """
        
        return context.strip()

# Usage with pattern tracking
advanced_memory = AdvancedMemorySystem()

conversation = ConversationChain(
    llm=OpenAI(temperature=0.7),
    memory=advanced_memory.combined_memory,
    verbose=True
)

# Simulate a conversation with pattern tracking
user_id = "sarah_123"

# Track that Sarah prefers detailed responses
advanced_memory.track_interaction_pattern(user_id, {
    'response_length_preference': 'detailed',
    'session_duration': 15.5,
    'topics': ['machine learning', 'data science', 'python'],
    'complexity_preference': 'advanced'
})

# Get personalized context
personal_context = advanced_memory.get_personalized_context(user_id)
print(f"Sarah's interaction style: {personal_context}")

# Now conversations can be tailored to Sarah's preferences
response = conversation.predict(
    input=f"{personal_context}\n\nUser question: How do I optimize my neural network?"
)

-----------------------------------------------------------------------------------------------------------------
### Memory Compression and Optimization
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

class IntelligentMemoryCompressor:
    def __init__(self):
        self.compression_rules = {
            'importance_threshold': 0.6,
            'recency_weight': 0.3,
            'frequency_weight': 0.4,
            'user_engagement_weight': 0.3
        }
    
    def calculate_memory_importance(self, memory: Dict[str, Any]) -> float:
        """Calculate how important a memory is for retention"""
        
        # Recency score (newer = more important)
        memory_date = datetime.fromisoformat(memory['timestamp'])
        days_old = (datetime.now() - memory_date).days
        recency_score = max(0, 1 - (days_old / 30))  # Decay over 30 days
        
        # Frequency score (mentioned more = more important)
        frequency_score = min(1.0, memory.get('mention_count', 1) / 10)
        
        # User engagement score (longer responses = more important)
        engagement_score = min(1.0, len(memory.get('content', '')) / 500)
        
        # Keywords that indicate importance
        important_keywords = ['problem', 'project', 'deadline', 'important', 'urgent', 'remember']
        keyword_score = sum(1 for keyword in important_keywords 
                          if keyword in memory.get('content', '').lower()) / len(important_keywords)
        
        # Weighted combination
        rules = self.compression_rules
        total_score = (
            recency_score * rules['recency_weight'] +
            frequency_score * rules['frequency_weight'] +
            engagement_score * rules['user_engagement_weight'] +
            keyword_score * 0.2
        )
        
        return min(1.0, total_score)
    
    def compress_memories(self, memories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Intelligently compress memory list"""
        
        # Score all memories
        scored_memories = []
        for memory in memories:
            importance = self.calculate_memory_importance(memory)
            scored_memories.append((importance, memory))
        
        # Sort by importance
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Keep top 70% of important memories
        keep_count = int(len(scored_memories) * 0.7)
        important_memories = [memory for _, memory in scored_memories[:keep_count]]
        
        # Summarize the rest
        less_important = [memory for _, memory in scored_memories[keep_count:]]
        if less_important:
            summary = self.create_memory_summary(less_important)
            important_memories.append({
                'type': 'summary',
                'content': summary,
                'timestamp': datetime.now().isoformat(),
                'original_count': len(less_important)
            })
        
        return important_memories
    
    def create_memory_summary(self, memories: List[Dict[str, Any]]) -> str:
        """Create a summary of multiple memories"""
        # Simple summarization - in production you'd use an LLM
        topics = {}
        for memory in memories:
            content = memory.get('content', '')
            # Extract topics (simplified)
            words = content.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    topics[word] = topics.get(word, 0) + 1
        
        # Get most common topics
        common_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        topic_list = [topic for topic, count in common_topics]
        
        return f"Summary of {len(memories)} conversations covering: {', '.join(topic_list)}"

# Usage example
compressor = IntelligentMemoryCompressor()

# Simulate a large memory collection
large_memory_collection = [
    {
        'content': 'User asked about machine learning project deadline',
        'timestamp': (datetime.now() - timedelta(days=5)).isoformat(),
        'mention_count': 3
    },
    {
        'content': 'User mentioned they like coffee',
        'timestamp': (datetime.now() - timedelta(days=20)).isoformat(),
        'mention_count': 1
    },
    # ... many more memories
]

# Compress intelligently
optimized_memories = compressor.compress_memories(large_memory_collection)
print(f"Compressed {len(large_memory_collection)} memories to {len(optimized_memories)}")

-----------------------------------------------------------------------------------------------------------------
### Agno’s Production-Grade Memory Architecture
from agno import Agent
from agno.memory import VectorMemory
import asyncio
from typing import Dict, List, Optional
import time

class ProductionMemoryManager:
    def __init__(self):
        self.user_agents: Dict[str, Agent] = {}
        self.memory_analytics = {
            'total_memories': 0,
            'average_retrieval_time': 0.0,
            'memory_hit_rate': 0.0
        }
    
    async def get_or_create_agent(self, user_id: str) -> Agent:
        """Get existing agent or create new one with optimized memory"""
        
        if user_id not in self.user_agents:
            # Create vector memory with production settings
            memory = VectorMemory(
                collection_name=f"prod_user_{user_id}",
                embeddings_model="text-embedding-3-large",
                distance_metric="cosine",
                # Production optimizations
                batch_size=50,
                index_type="IVF",  # Faster for large datasets
                ef_construction=200,  # Balance between speed and accuracy
                max_connections=16
            )
            
            # Create agent with production-grade instructions
            agent = Agent(
                model="gpt-4-turbo",
                memory=memory,
                instructions=f"""
                You are an advanced AI assistant with sophisticated memory capabilities.
                
                Memory Usage Guidelines:
                - Always reference relevant past conversations when appropriate
                - Build on previous knowledge rather than repeating information
                - Adapt your communication style based on user interaction patterns
                - Prioritize recent and frequently mentioned topics
                - If you're unsure about a memory, ask for clarification rather than guessing
                
                User ID: {user_id}
                """,
                # Production settings
                temperature=0.7,
                max_retries=3,
                timeout=30
            )
            
            self.user_agents[user_id] = agent
        
        return self.user_agents[user_id]
    
    async def chat_with_analytics(self, user_id: str, message: str) -> Dict[str, Any]:
        """Chat with full analytics and memory optimization"""
        start_time = time.time()
        
        agent = await self.get_or_create_agent(user_id)
        
        # Enhanced memory retrieval with context scoring
        relevant_memories = await self._get_contextual_memories(agent, message)
        
        # Generate response
        response = await agent.run(message)
        
        # Calculate analytics
        processing_time = time.time() - start_time
        self._update_analytics(processing_time, len(relevant_memories))
        
        return {
            'response': response,
            'processing_time': processing_time,
            'memories_used': len(relevant_memories),
            'memory_confidence': await self._calculate_memory_confidence(relevant_memories, message)
        }
    
    async def _get_contextual_memories(self, agent: Agent, query: str) -> List[Dict]:
        """Get memories with enhanced context scoring"""
        # Get initial memory results
        memories = await agent.memory.search(
            query=query,
            limit=10,
            include_metadata=True
        )
        
        # Score memories based on multiple factors
        scored_memories = []
        for memory in memories:
            # Base similarity score from vector search
            base_score = memory.get('score', 0.0)
            
            # Recency boost
            recency_boost = self._calculate_recency_boost(memory.get('metadata', {}))
            
            # Interaction frequency boost
            freq_boost = self._calculate_frequency_boost(memory.get('metadata', {}))
            
            # Final score
            final_score = base_score + recency_boost + freq_boost
            scored_memories.append((final_score, memory))
        
        # Return top scored memories
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:5]]
    
    def _calculate_recency_boost(self, metadata: Dict) -> float:
        """Give boost to more recent memories"""
        if 'timestamp' not in metadata:
            return 0.0
        
        # Simple recency calculation
        try:
            memory_time = datetime.fromisoformat(metadata['timestamp'])
            hours_old = (datetime.now() - memory_time).total_seconds() / 3600
            # Boost decreases over time
            return max(0, 0.1 - (hours_old / 240))  # 10-day decay
        except:
            return 0.0
    
    def _calculate_frequency_boost(self, metadata: Dict) -> float:
        """Give boost to frequently accessed memories"""
        access_count = metadata.get('access_count', 0)
        return min(0.05, access_count * 0.01)  # Max 5% boost
    
    async def _calculate_memory_confidence(self, memories: List[Dict], query: str) -> float:
        """Calculate confidence in memory retrieval quality"""
        if not memories:
            return 0.0
        
        # Average of memory scores
        scores = [m.get('score', 0.0) for m in memories]
        avg_score = sum(scores) / len(scores)
        
        # Adjust for number of memories found
        count_factor = min(1.0, len(memories) / 3)  # Optimal around 3 memories
        
        return avg_score * count_factor
    
    def _update_analytics(self, processing_time: float, memories_used: int):
        """Update system analytics"""
        self.memory_analytics['total_memories'] += memories_used
        
        # Update average processing time
        current_avg = self.memory_analytics['average_retrieval_time']
        self.memory_analytics['average_retrieval_time'] = (current_avg * 0.9) + (processing_time * 0.1)
        
        # Update hit rate (memories found vs queries)
        hit = 1.0 if memories_used > 0 else 0.0
        current_hit_rate = self.memory_analytics['memory_hit_rate']
        self.memory_analytics['memory_hit_rate'] = (current_hit_rate * 0.9) + (hit * 0.1)
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get memory system health metrics"""
        return {
            'active_users': len(self.user_agents),
            'total_memories_retrieved': self.memory_analytics['total_memories'],
            'average_response_time': self.memory_analytics['average_retrieval_time'],
            'memory_hit_rate': self.memory_analytics['memory_hit_rate'],
            'system_status': 'healthy' if self.memory_analytics['memory_hit_rate'] > 0.7 else 'degraded'
        }

# Production usage example
async def production_example():
    memory_manager = ProductionMemoryManager()
    
    # Simulate multiple users
    users = ['alice_123', 'bob_456', 'charlie_789']
    
    for user_id in users:
        # Each user has their own memory-enabled agent
        result = await memory_manager.chat_with_analytics(
            user_id=user_id,
            message="I'm working on a new project and need some advice"
        )
        
        print(f"User {user_id}:")
        print(f"  Response time: {result['processing_time']:.2f}s")
        print(f"  Memories used: {result['memories_used']}")
        print(f"  Confidence: {result['memory_confidence']:.2f}")
        print()
    
    # Check system health
    health = await memory_manager.get_system_health()
    print(f"System Health: {health}")

# Run the production example
asyncio.run(production_example())

-----------------------------------------------------------------------------------------------------------------
### Custom Memory Stores for Specific Use Cases
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import sqlite3
import json
from datetime import datetime

class CustomMemoryStore(BaseModel):
    """Custom memory store for specialized applications"""
    
    class Config:
        arbitrary_types_allowed = True
    
    db_path: str = "custom_memory.db"
    
    def __init__(self, **data):
        super().__init__(**data)
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for memory storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables for different types of memories
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                message TEXT,
                response TEXT,
                timestamp TEXT,
                importance_score REAL,
                tags TEXT  -- JSON array of tags
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                communication_style TEXT,
                expertise_level TEXT,
                interests TEXT,  -- JSON array
                last_updated TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS learning_patterns (
                id INTEGER PRIMARY KEY,
                user_id TEXT,
                pattern_type TEXT,
                pattern_data TEXT,  -- JSON
                confidence_score REAL,
                last_reinforced TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def store_conversation(self, user_id: str, message: str, response: str, 
                                importance: float = 0.5, tags: List[str] = None):
        """Store a conversation with metadata"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO conversations (user_id, message, response, timestamp, importance_score, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            user_id, 
            message, 
            response, 
            datetime.now().isoformat(),
            importance,
            json.dumps(tags or [])
        ))
        
        conn.commit()
        conn.close()
    
    async def learn_user_pattern(self, user_id: str, pattern_type: str, pattern_data: Dict[str, Any]):
        """Store learned patterns about user behavior"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if pattern already exists
        cursor.execute('''
            SELECT confidence_score FROM learning_patterns 
            WHERE user_id = ? AND pattern_type = ?
        ''', (user_id, pattern_type))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing pattern with reinforcement
            new_confidence = min(1.0, existing[0] + 0.1)
            cursor.execute('''
                UPDATE learning_patterns 
                SET pattern_data = ?, confidence_score = ?, last_reinforced = ?
                WHERE user_id = ? AND pattern_type = ?
            ''', (
                json.dumps(pattern_data),
                new_confidence,
                datetime.now().isoformat(),
                user_id,
                pattern_type
            ))
        else:
            # Create new pattern
            cursor.execute('''
                INSERT INTO learning_patterns (user_id, pattern_type, pattern_data, confidence_score, last_reinforced)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                user_id,
                pattern_type,
                json.dumps(pattern_data),
                0.5,
                datetime.now().isoformat()
            ))
        
        conn.commit()
        conn.close()
    
    async def get_adaptive_context(self, user_id: str, current_query: str) -> str:
        """Generate adaptive context based on learned patterns"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get user preferences
        cursor.execute('SELECT * FROM user_preferences WHERE user_id = ?', (user_id,))
        prefs = cursor.fetchone()
        
        # Get learned patterns
        cursor.execute('''
            SELECT pattern_type, pattern_data, confidence_score 
            FROM learning_patterns 
            WHERE user_id = ? AND confidence_score > 0.6
            ORDER BY confidence_score DESC
        ''', (user_id,))
        patterns = cursor.fetchall()
        
        # Get relevant conversations
        cursor.execute('''
            SELECT message, response, importance_score, tags
            FROM conversations 
            WHERE user_id = ?
            ORDER BY importance_score DESC, timestamp DESC
            LIMIT 5
        ''', (user_id,))
        conversations = cursor.fetchall()
        
        conn.close()
        
        # Build adaptive context
        context_parts = []
        
        if prefs:
            context_parts.append(f"User communication style: {prefs[1]}")
            context_parts.append(f"Expertise level: {prefs[2]}")
        
        if patterns:
            context_parts.append("Learned patterns:")
            for pattern_type, pattern_data, confidence in patterns[:3]:
                data = json.loads(pattern_data)
                context_parts.append(f"  - {pattern_type}: {data} (confidence: {confidence:.2f})")
        
        if conversations:
            context_parts.append("Relevant past conversations:")
            for msg, resp, importance, tags in conversations[:2]:
                context_parts.append(f"  - User asked about: {msg[:100]}...")
        
        return "\n".join(context_parts)

class AdaptiveMemoryAgent:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memory_store = CustomMemoryStore()
        self.agent = Agent('openai:gpt-4')
    
    async def chat(self, message: str) -> str:
        """Chat with adaptive memory learning"""
        
        # Get adaptive context
        context = await self.memory_store.get_adaptive_context(self.user_id, message)
        
        # Generate response with context
        full_prompt = f"Context about this user:\n{context}\n\nUser message: {message}"
        response = await self.agent.run(full_prompt)
        
        # Analyze and learn from this interaction
        await self._learn_from_interaction(message, response.data)
        
        # Store conversation
        importance = await self._calculate_importance(message)
        await self.memory_store.store_conversation(
            self.user_id, 
            message, 
            response.data, 
            importance
        )
        
        return response.data
    
    async def _learn_from_interaction(self, message: str, response: str):
        """Learn patterns from user interaction"""
        
        # Learn about question complexity preference
        message_complexity = 'high' if len(message.split()) > 20 else 'medium' if len(message.split()) > 10 else 'low'
        response_length = len(response.split())
        
        await self.memory_store.learn_user_pattern(
            self.user_id,
            "question_complexity",
            {"typical_complexity": message_complexity, "preferred_response_length": response_length}
        )
        
        # Learn about topic interests
        if any(tech_word in message.lower() for tech_word in ['python', 'programming', 'code', 'algorithm']):
            await self.memory_store.learn_user_pattern(
                self.user_id,
                "topic_interest",
                {"category": "programming", "engagement_level": "high"}
            )
    
    async def _calculate_importance(self, message: str) -> float:
        """Calculate importance of this conversation"""
        # Simple importance calculation
        importance_keywords = ['important', 'urgent', 'problem', 'help', 'project', 'deadline']
        
        score = 0.3  # Base score
        message_lower = message.lower()
        
        for keyword in importance_keywords:
            if keyword in message_lower:
                score += 0.2
        
        # Question length might indicate complexity/importance
        if len(message.split()) > 15:
            score += 0.1
        
        return min(1.0, score)

# Usage example
async def custom_memory_example():
    agent = AdaptiveMemoryAgent("dev_user_001")
    
    # Conversation that teaches the system
    responses = []
    
    responses.append(await agent.chat("I'm working on a Python project and need help with algorithms"))
    responses.append(await agent.chat("Can you explain binary search in detail?"))
    responses.append(await agent.chat("What's the time complexity of quicksort?"))
    responses.append(await agent.chat("I prefer detailed explanations with examples"))
    
    # Later conversation - system should adapt
    adapted_response = await agent.chat("How does merge sort work?")
    
    print("Adaptive response based on learned patterns:")
    print(adapted_response)

# Run the example
asyncio.run(custom_memory_example())

------------------------------------------------------------------------------------
## Example 
import time
import asyncio
from dataclasses import dataclass
from typing import Dict, List
import psutil
import logging

@dataclass
class MemoryMetrics:
    """Memory system performance metrics"""
    retrieval_time_avg: float = 0.0
    storage_time_avg: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0
    total_operations: int = 0
    error_rate: float = 0.0

class MemoryPerformanceMonitor:
    def __init__(self):
        self.metrics = MemoryMetrics()
        self.operation_times: List[float] = []
        self.error_count = 0
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('MemoryMonitor')
    
    def time_operation(self, operation_type: str):
        """Decorator to time memory operations"""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    
                    # Record successful operation
                    operation_time = time.time() - start_time
                    self._record_operation(operation_type, operation_time, success=True)
                    
                    return result
                    
                except Exception as e:
                    # Record failed operation
                    operation_time = time.time() - start_time
                    self._record_operation(operation_type, operation_time, success=False)
                    self.logger.error(f"Memory operation failed: {e}")
                    raise
                    
            return wrapper
        return decorator
    
    def _record_operation(self, operation_type: str, duration: float, success: bool):
        """Record operation metrics"""
        self.metrics.total_operations += 1
        self.operation_times.append(duration)
        
        if not success:
            self.error_count += 1
        
        # Update running averages
        if operation_type == 'retrieval':
            self.metrics.retrieval_time_avg = self._update_average(
                self.metrics.retrieval_time_avg, duration
            )
        elif operation_type == 'storage':
            self.metrics.storage_time_avg = self._update_average(
                self.metrics.storage_time_avg, duration
            )
        
        # Update error rate
        self.metrics.error_rate = self.error_count / self.metrics.total_operations
        
        # Update memory usage
        self.metrics.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
    
    def _update_average(self, current_avg: float, new_value: float) -> float:
        """Update running average with exponential decay"""
        return (current_avg * 0.9) + (new_value * 0.1)
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not self.operation_times:
            return {"status": "no_data", "message": "No operations recorded yet"}
        
        # Calculate percentiles
        sorted_times = sorted(self.operation_times)
        p50 = sorted_times[len(sorted_times) // 2]
        p95 = sorted_times[int(len(sorted_times) * 0.95)]
        p99 = sorted_times[int(len(sorted_times) * 0.99)]
        
        return {
            "status": "healthy" if self.metrics.error_rate < 0.05 else "degraded",
            "total_operations": self.metrics.total_operations,
            "error_rate_percent": self.metrics.error_rate * 100,
            "memory_usage_mb": self.metrics.memory_usage_mb,
            "performance": {
                "avg_retrieval_time_ms": self.metrics.retrieval_time_avg * 1000,
                "avg_storage_time_ms": self.metrics.storage_time_avg * 1000,
                "p50_ms": p50 * 1000,
                "p95_ms": p95 * 1000,
                "p99_ms": p99 * 1000
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        if self.metrics.retrieval_time_avg > 0.5:
            recommendations.append("Consider adding memory caching or indexing")
        
        if self.metrics.memory_usage_mb > 500:
            recommendations.append("Memory usage is high - consider implementing compression")
        
        if self.metrics.error_rate > 0.02:
            recommendations.append("Error rate is elevated - check memory system stability")
        
        if len(self.operation_times) > 1000 and max(self.operation_times) > 2.0:
            recommendations.append("Some operations are very slow - investigate bottlenecks")
        
        return recommendations or ["Memory system performance looks good!"]

# Integration with production memory system
class MonitoredMemoryAgent:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.monitor = MemoryPerformanceMonitor()
        self.agent = Agent('openai:gpt-4')
        self.memory_cache = {}  # Simple in-memory cache
    
    @property
    def time_retrieval(self):
        return self.monitor.time_operation('retrieval')
    
    @property 
    def time_storage(self):
        return self.monitor.time_operation('storage')
    
    @time_retrieval
    async def retrieve_memories(self, query: str) -> List[Dict]:
        """Retrieve memories with performance monitoring"""
        
        # Check cache first
        cache_key = f"{self.user_id}:{hash(query)}"
        if cache_key in self.memory_cache:
            return self.memory_cache[cache_key]
        
        # Simulate memory retrieval (replace with actual implementation)
        await asyncio.sleep(0.1)  # Simulate database query
        
        memories = [
            {"content": "Sample memory", "score": 0.8},
            {"content": "Another memory", "score": 0.6}
        ]
        
        # Cache results
        self.memory_cache[cache_key] = memories
        
        return memories
    
    @time_storage
    async def store_memory(self, content: str) -> bool:
        """Store memory with performance monitoring"""
        
        # Simulate memory storage
        await asyncio.sleep(0.05)  # Simulate database write
        
        return True
    
    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return await self.monitor.get_performance_report()

# Usage example
async def monitoring_example():
    agent = MonitoredMemoryAgent("perf_test_user")
    
    # Simulate memory operations
    for i in range(50):
        await agent.retrieve_memories(f"query {i}")
        await agent.store_memory(f"memory content {i}")
    
    # Get performance report
    report = await agent.get_performance_metrics()
    print("Memory System Performance Report:")
    print(f"Status: {report['status']}")
    print(f"Total Operations: {report['total_operations']}")
    print(f"Error Rate: {report['error_rate_percent']:.2f}%")
    print(f"Memory Usage: {report['memory_usage_mb']:.1f} MB")
    print(f"Average Retrieval Time: {report['performance']['avg_retrieval_time_ms']:.1f}ms")
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")

asyncio.run(monitoring_example())
