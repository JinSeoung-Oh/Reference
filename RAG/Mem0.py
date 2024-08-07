"""
From https://ai.gopubby.com/mem0-is-this-the-future-of-ai-memory-management-1e228dc8220a

1. Overview of Mem0
   Mem0 is an advanced memory management system for AI applications that functions like a library,
   allowing immediate recall and reference to all previously encountered data. 
   It creates a dedicated memory space in a computer to facilitate the development of personalized,
   reliable, and cost-effective AI applications such as chatbots, virtual assistants, or AI agents.

2. Key Features of Mem0
   -1. Personalization:
       - Mem0 enables AI to provide personalized responses by recalling past interactions.
       - Example: A virtual assistant remembering a user’s favorite coffee order from last week.
   -2. Reliability:
       - Ensures consistent and accurate responses by referencing a user’s history.
       - Example: A customer support chatbot using Mem0 can draw on a user’s previous support interactions.
   -3. Cost-Effectiveness:
       - Reduces the need for frequent queries to expensive Large Language Models (LLMs) by storing and reusing information.
       - Example: An AI writing assistant can recall previous edits without re-querying the LLM.
   -4. Engagement:
       - Enhances user engagement by personalizing interactions.
       - Example: An educational chatbot remembers a student’s progress, tailoring lessons accordingly.
   -5. Long-Term Memory:
       - Retains information across sessions, allowing for a deeper understanding of user needs.
       - Example: A health management AI remembers a patient’s medical history over multiple visits.

3. Comparison: Mem0 vs. RAG and Conversation Context Management
   1. Mem0 vs. Retrieval-Augmented Generation (RAG)
      -1. Static vs. Dynamic: RAG is known for static knowledge base searches, while Mem0 offers real-time updates and dynamic searches.
      -2. Applications: Mem0 is ideal for applications that require dynamic context handling, such as AI chatbot conversations.
   2. Traditional Methods of Conversation Context Handling
      -1. Session Management:
          Manages conversation context within a session.
      -2. Database Storage:
         - Pros: Provides long-term, scalable storage for complex interactions and persistent conversation tracking.
         - Cons: Requires setup and management, which can add complexity and latency.
   3. HTML Session Storage:
      - Pros: Easy to implement and suitable for temporary data storage in the browser, reducing server load.
      - Cons: Limited capacity and session duration, unsuitable for long-term or complex functionality.
   4. Advantages of Mem0
      -1. Real-Time Updates:
          Offers real-time updates and searches, ensuring access to the most recent and relevant information.
      -2. Flexible Context Handling:
          Dynamically adjusts conversation context, using memory search results as part of the prompt, enhancing efficiency.
      -3. Personalization and Reliability:
          Maintains a dedicated memory space for personalized responses, reflecting user history and preferences.
      -4. Cost-Effectiveness:
          Reduces the need for frequent calls to LLMs, enhancing cost-effectiveness.

"Mem0 Won’t Replace RAG, But It Enhances It"
"""
import openai
import os
from mem0 import Memory
from multion.client import MultiOn

# Configure API keys
OPENAI_API_KEY = 'your-openai-api-key'
MULTION_API_KEY = 'your-multion-api-key'
USER_ID = "kelvin"

# Set OpenAI API key
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
memory = Memory()
multion = MultiOn(api_key=MULTION_API_KEY)

# Store medical history record
memory.add("Charlie has Type 2 diabetes and takes metformin daily.", user_id=USER_ID)

# Query related memory
command = "How to manage diabetes?"
related_memories = memory.search(command, user_id=USER_ID)
related_memories_text = '\n'.join(mem['text'] for mem in related_memories)

# Create a prompt with relevant memories
prompt0 = f"{command}\nMy past memories: {related_memories_text}"

openai.organization = "your-openai-organization"
openai.api_key = os.getenv('OPENAI_API_KEY')

# Generate response using OpenAI
client = openai.OpenAI()
completion = client.Completion.create(
  model="gpt-3.5-turbo-instruct", 
  prompt=prompt0,
  max_tokens=150,
  temperature=0
)

print(completion.choices[0].text)


##### AI Automatic Job Interview System
!pip install openai mem0ai

from openai import OpenAI
from mem0 import Memory
import os

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = 'your-openai-api-key'

class JobInterviewAIAgent:
    def __init__(self):
        """
        Initialize the JobInterviewAIAgent with memory configuration and OpenAI client.
        """
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "host": "localhost",
                    "port": 6333,
                }
            },
        }
        self.memory = Memory.from_config(config)
        self.client = OpenAI()
        self.app_id = "job-interview"

    def handle_query(self, query, user_id=None):
        """
        Handle a candidate's query and store the relevant information in memory.
        """
        # Start a streaming chat completion request to the AI
        stream = self.client.ChatCompletion.create(
            model="gpt-4",
            stream=True,
            messages=[
                {"role": "system", "content": "You are an AI job interview agent."},
                {"role": "user", "content": query}
            ]
        )
        # Store the query in memory
        self.memory.add(query, user_id=user_id, metadata={"app_id": self.app_id})

        # Print the response from the AI in real-time
        for chunk in stream:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")

    def get_memories(self, user_id=None):
        """
        Retrieve all memories associated with the given candidate ID.
        """
        return self.memory.get_all(user_id=user_id)

# Instantiate the JobInterviewAIAgent
interview_agent = JobInterviewAIAgent()

# Define a candidate ID
candidate_id = "john_doe"

# Handle a candidate query
interview_agent.handle_query("Can you tell me about the company culture?", user_id=candidate_id)

# Fetching Memories
memories = interview_agent.get_memories(user_id=candidate_id)
for memory in memories:
    print(memory['text'])







