### From https://levelup.gitconnected.com/implementing-9-techniques-to-optimize-ai-agent-memory-67d813e3d796
"""
1. Sequential Memory simply appends every user–AI exchange into one ever‑growing context and sends the entire history to the model at each turn. 
   It’s trivial to implement and guarantees no loss of detail, but as the conversation lengthens the token count—and therefore cost and latency—skyrockets, 
   quickly hitting API limits and becoming impractical.
2. Sliding Window Memory keeps only the most recent N turns in context, automatically discarding older ones. 
   This caps the prompt size and makes both compute and cost predictable, but any information pushed out of the window is permanently forgotten, 
   so the model can no longer recall earlier details.
3. Summarization Memory periodically asks the LLM to condense the accumulated conversation into a short running summary, then clears the buffer. 
   By combining that summary with only the most recent messages, it dramatically cuts token usage while preserving key points. 
   Its downside is that an imperfect summary can omit critical facts—especially numeric or highly specific details—so accuracy depends heavily on 
   prompt design and the model’s summarization quality.
4. Retrieval‑Based Memory (RAG) treats each turn as a separate document, embeds it into a vector store, 
   and on every query retrieves only the top k semantically relevant documents. 
   This delivers highly focused context, slashes token counts, and scales to lengthy dialogues, but requires building and maintaining an embedding pipeline,
   vector index (e.g. FAISS), and fast similarity search infrastructure.
5. Memory‑Augmented Strategies layer a short‑term buffer (e.g. sliding window) with a parallel long‑term list of “memory tokens” that the model itself
   flags—via extra LLM calls—as important facts or preferences. 
   This ensures that must‑remember details (allergies, user IDs, etc.) survive indefinitely, but each fact‑extraction call incurs additional cost and latency.
6. Hierarchical Memory combines both worlds: every turn goes into fast working memory, but any message containing promotion keywords 
   (“remember,” “always,” “never,” “allergic,” etc.) is also added to a retrieval‑based long‑term store. 
   Queries pull from both layers, giving the model both conversational flow and deep facts. 
   It offers a balanced trade‑off, but you must carefully tune keywords and manage two systems in tandem.
7. Graph‑Based Memory uses the LLM as a triple extractor to convert conversation into Subject–Relation–Object edges in a knowledge graph (via NetworkX). 
   The model can then answer relationship queries by traversing that graph. 
   This empowers complex, structured reasoning, but demands accurate triple extraction and a more complex graph‑update and query workflow.
8. Compression & Consolidation Memory makes an LLM compress each turn into a single, ultra‑concise factual statement, stripping all fluff. 
   The result is an extremely token‑efficient list of core facts, but you lose conversational nuance and supporting detail, 
   making it unsuitable when tone or context matters.
9. OS‑Like Memory Management simulates RAM vs. disk: a small active buffer holds the newest turns, paging out older turns into passive storage when it’s full, 
   and paging them back in on a keyword match “page fault.” 
   This gives virtually unbounded memory capacity with a bounded active context, but requires designing eviction and retrieval policies and 
   can add complexity in handling page‑in triggers.

Which to choose?
-a. Simple, short‐lived bots: Sequential or Sliding Window.
-b. Long, creative dialogues: Summarization.
-c. Accurate long‐term recall: Retrieval‑Based Memory (RAG).
-d. Personal assistants needing key‐fact persistence: Memory‑Augmented or Hierarchical Memory.
-e. Expert systems requiring relationship reasoning: Graph‑Based Memory.
-f. Tight token budgets with factual recall: Compression Memory.
-g. Massive memory with minimal active context: OS‑Like Memory.

In practice, hybridizing these—for example, a hierarchical system that promotes to both a vector store and a knowledge graph—often delivers
the best balance of cost, performance, and reliability.
"""
------------------------------------------------------------------------------------------------------------
### Setting up the Environment
!pip install openai numpy faiss-cpu networkx tiktoken

# Import necessary libraries
import os
from openai import OpenAI

# Define the API key for authentication.
API_KEY = "YOUR_LLM_API_KEY"

# Define the base URL for the API endpoint.
BASE_URL = "https://api.studio.nebius.com/v1/"

# Initialize the OpenAI client with the specified base URL and API key.
client = OpenAI(
    base_url=BASE_URL,
    api_key=API_KEY
)

# Print a confirmation message to indicate successful client setup.
print("OpenAI client configured successfully.")

# Import additional libraries for functionality.
import tiktoken
import time

# --- Model Configuration ---
# Define the specific models to be used for generation and embedding tasks.
# These are hardcoded for this lab but could be loaded from a config file.
GENERATION_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"

------------------------------------------------------------------------------------------------------------
### Creating Helper Functions
def generate_text(system_prompt: str, user_prompt: str) -> str:
    """
    Calls the LLM API to generate a text response.
    
    Args:
        system_prompt: The instruction that defines the AI's role and behavior.
        user_prompt: The user's input to which the AI should respond.
        
    Returns:
        The generated text content from the AI, or an error message.
    """
    # Create a chat completion request to the configured client.
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    # Extract and return the content of the AI's message.
    return response.choices[0].message.content

def generate_embedding(text: str) -> list[float]:
    """
    Generates a numerical embedding for a given text string using the embedding model.
    
    Args:
        text: The input string to be converted into an embedding.
        
    Returns:
        A list of floats representing the embedding vector, or an empty list on error.
    """
    # Create an embedding request to the configured client.
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    # Extract and return the embedding vector from the response data.
    return response.data[0].embedding

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given string using the pre-loaded tokenizer.
    
    Args:
        text: The string to be tokenized.
        
    Returns:
        The integer count of tokens.
    """
    # The `encode` method converts the string into a list of token IDs.
    # The length of this list is the token count.
    return len(tokenizer.encode(text))

------------------------------------------------------------------------------------------------------------
### Creating Foundational Agent and Memory Class

# --- Abstract Base Class for Memory Strategies ---
# This class defines the 'contract' that all memory strategies must follow.
# By using an Abstract Base Class (ABC), we ensure that any memory implementation
# we create will have the same core methods (add_message, get_context, clear),
# allowing them to be interchangeably plugged into the AIAgent.
class BaseMemoryStrategy(abc.ABC):
    """Abstract Base Class for all memory strategies."""
    
    @abc.abstractmethod
    def add_message(self, user_input: str, ai_response: str):
        """
        An abstract method that must be implemented by subclasses.
        It's responsible for adding a new user-AI interaction to the memory store.
        """
        pass

    @abc.abstractmethod
    def get_context(self, query: str) -> str:
        """
        An abstract method that must be implemented by subclasses.
        It retrieves and formats the relevant context from memory to be sent to the LLM.
        The 'query' parameter allows some strategies (like retrieval) to fetch context
        that is specifically relevant to the user's latest input.
        """
        pass

    @abc.abstractmethod
    def clear(self):
        """
        An abstract method that must be implemented by subclasses.
        It provides a way to reset the memory, which is useful for starting new conversations.
        """
        pass

# --- The Core AI Agent ---
# This class orchestrates the entire conversation flow. It is initialized with a
# specific memory strategy and uses it to manage the conversation's context.
class AIAgent:
    """The main AI Agent class, designed to work with any memory strategy."""
    
    def __init__(self, memory_strategy: BaseMemoryStrategy, system_prompt: str = "You are a helpful AI assistant."):
        """
        Initializes the agent.
        
        Args:
            memory_strategy: An instance of a class that inherits from BaseMemoryStrategy.
                             This determines how the agent will remember the conversation.
            system_prompt: The initial instruction given to the LLM to define its persona and task.
        """
        self.memory = memory_strategy
        self.system_prompt = system_prompt
        print(f"Agent initialized with {type(memory_strategy).__name__}.")

    def chat(self, user_input: str):
        """
        Handles a single turn of the conversation.
        
        Args:
            user_input: The latest message from the user.
        """
        print(f"\n{'='*25} NEW INTERACTION {'='*25}")
        print(f"User > {user_input}")
        
        # Step 1: Retrieve context from the agent's memory strategy.
        # This is where the specific memory logic (e.g., sequential, retrieval) is executed.
        start_time = time.time()
        context = self.memory.get_context(query=user_input)
        retrieval_time = time.time() - start_time
        
        # Step 2: Construct the full prompt for the LLM.
        # This combines the retrieved historical context with the user's current request.
        full_user_prompt = f"### MEMORY CONTEXT\n{context}\n\n### CURRENT REQUEST\n{user_input}"
        
        # Step 3: Provide detailed debug information.
        # This is crucial for understanding how the memory strategy affects the prompt size and cost.
        prompt_tokens = count_tokens(self.system_prompt + full_user_prompt)
        print("\n--- Agent Debug Info ---")
        print(f"Memory Retrieval Time: {retrieval_time:.4f} seconds")
        print(f"Estimated Prompt Tokens: {prompt_tokens}")
        print(f"\n[Full Prompt Sent to LLM]:\n---\nSYSTEM: {self.system_prompt}\nUSER: {full_user_prompt}\n---")
        
        # Step 4: Call the LLM to get a response.
        # The LLM uses the system prompt and the combined user prompt (context + new query) to generate a reply.
        start_time = time.time()
        ai_response = generate_text(self.system_prompt, full_user_prompt)
        generation_time = time.time() - start_time
        
        # Step 5: Update the memory with the latest interaction.
        # This ensures the current turn is available for future context retrieval.
        self.memory.add_message(user_input, ai_response)
        
        # Step 6: Display the AI's response and performance metrics.
        print(f"\nAgent > {ai_response}")
        print(f"(LLM Generation Time: {generation_time:.4f} seconds)")
        print(f"{'='*70}")

------------------------------------------------------------------------------------------------------------
### 1. Sequential Optimization Approach
# --- Strategy 1: Sequential (Keep-It-All) Memory ---
# This is the most basic memory strategy. It stores the entire conversation
# history in a simple list. While it provides perfect recall, it is not scalable
# as the context sent to the LLM grows with every turn, quickly becoming expensive
# and hitting token limits.
class SequentialMemory(BaseMemoryStrategy):
    def __init__(self):
        """Initializes the memory with an empty list to store conversation history."""
        self.history = []

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a new user-AI interaction to the history.
        Each interaction is stored as two dictionary entries in the list.
        """
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": ai_response})

    def get_context(self, query: str) -> str:
        """
        Retrieves the entire conversation history and formats it into a single
        string to be used as context for the LLM. The 'query' parameter is ignored
        as this strategy always returns the full history.
        """
        # Join all messages into a single newline-separated string.
        return "\n".join([f"{turn['role'].capitalize()}: {turn['content']}" for turn in self.history])

    def clear(self):
        """Resets the conversation history by clearing the list."""
        self.history = []
        print("Sequential memory cleared.")

# Initialize and run the agent
# Create an instance of our SequentialMemory strategy.
sequential_memory = SequentialMemory()
# Create an AIAgent and inject the sequential memory strategy into it.
agent = AIAgent(memory_strategy=sequential_memory)

------------------------------------------------------------------------------------------------------------
### 2. Sliding Window Approach
# --- Strategy 2: Sliding Window Memory ---
# This strategy keeps only the 'N' most recent turns of the conversation.
# It prevents the context from growing indefinitely, making it scalable and
# cost-effective, but at the cost of forgetting older information.
class SlidingWindowMemory(BaseMemoryStrategy):
    def __init__(self, window_size: int = 4): # window_size is number of turns (user + AI = 1 turn)
        """
        Initializes the memory with a deque of a fixed size.
        
        Args:
            window_size: The number of conversational turns to keep in memory.
                         A single turn consists of one user message and one AI response.
        """
        # A deque with 'maxlen' will automatically discard the oldest item
        # when a new item is added and the deque is full. This is the core
        # mechanism of the sliding window. We store turns, so maxlen is window_size.
        self.history = deque(maxlen=window_size)

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a new conversational turn to the history. If the deque is full,
        the oldest turn is automatically removed.
        """
        # Each turn (user input + AI response) is stored as a single element
        # in the deque. This makes it easy to manage the window size by turns.
        self.history.append([
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": ai_response}
        ])

    def get_context(self, query: str) -> str:
        """
        Retrieves the conversation history currently within the window and
        formats it into a single string. The 'query' parameter is ignored.
        """
        # Create a temporary list to hold the formatted messages.
        context_list = []
        # Iterate through each turn stored in the deque.
        for turn in self.history:
            # Iterate through the user and assistant messages within that turn.
            for message in turn:
                # Format the message and add it to our list.
                context_list.append(f"{message['role'].capitalize()}: {message['content']}")
        # Join all the formatted messages into a single string, separated by newlines.
        return "\n".join(context_list)
      
# Initialize with a small window size of 2 turns.
# This means the agent will only remember the last two user-AI interactions.
sliding_memory = SlidingWindowMemory(window_size=2)
# Create an AIAgent and inject the sliding window memory strategy.
agent = AIAgent(memory_strategy=sliding_memory)

------------------------------------------------------------------------------------------------------------
### 3. Summarization Based Optimization

# --- Strategy 3: Summarization Memory ---
# This strategy aims to manage long conversations by periodically summarizing them.
# It keeps a buffer of recent messages. When the buffer reaches a certain size,
# it uses an LLM call to consolidate the buffer's content with a running summary.
# This keeps the context size manageable while retaining the gist of the conversation.
# The main risk is information loss if the summary is not perfect.
class SummarizationMemory(BaseMemoryStrategy):
    def __init__(self, summary_threshold: int = 4): # Default: Summarize after 4 messages (2 turns)
        """
        Initializes the summarization memory.
        
        Args:
            summary_threshold: The number of messages (user + AI) to accumulate in the
                             buffer before triggering a summarization.
        """
        # Stores the continuously updated summary of the conversation so far.
        self.running_summary = ""
        # A temporary list to hold recent messages before they are summarized.
        self.buffer = []
        # The threshold that triggers the summarization process.
        self.summary_threshold = summary_threshold

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a new user-AI interaction to the buffer. If the buffer size
        reaches the threshold, it triggers the memory consolidation process.
        """
        # Append the latest user and AI messages to the temporary buffer.
        self.buffer.append({"role": "user", "content": user_input})
        self.buffer.append({"role": "assistant", "content": ai_response})

        # Check if the buffer has reached its capacity.
        if len(self.buffer) >= self.summary_threshold:
            # If so, call the method to summarize the buffer's contents.
            self._consolidate_memory()

    def _consolidate_memory(self):
        """
        Uses the LLM to summarize the contents of the buffer and merge it
        with the existing running summary.
        """
        print("\n--- [Memory Consolidation Triggered] ---")
        # Convert the list of buffered messages into a single formatted string.
        buffer_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.buffer])
        
        # Construct a specific prompt for the LLM to perform the summarization task.
        # It provides the existing summary and the new conversation text, asking for
        # a single, updated summary.
        summarization_prompt = (
            f"You are a summarization expert. Your task is to create a concise summary of a conversation. "
            f"Combine the 'Previous Summary' with the 'New Conversation' into a single, updated summary. "
            f"Capture all key facts, names, and decisions.\n\n"
            f"### Previous Summary:\n{self.running_summary}\n\n"
            f"### New Conversation:\n{buffer_text}\n\n"
            f"### Updated Summary:"
        )
        
        # Call the LLM with a specific system prompt to get the new summary.
        new_summary = generate_text("You are an expert summarization engine.", summarization_prompt)
        # Replace the old summary with the newly generated, consolidated one.
        self.running_summary = new_summary
        # Clear the buffer, as its contents have now been incorporated into the summary.
        self.buffer = [] 
        print(f"--- [New Summary: '{self.running_summary}'] ---")

    def get_context(self, query: str) -> str:
        """
        Constructs the context to be sent to the LLM. It combines the long-term
        running summary with the short-term buffer of recent messages.
        The 'query' parameter is ignored as this strategy provides a general context.
        """
        # Format the messages currently in the buffer.
        buffer_text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in self.buffer])
        # Return a combined context of the historical summary and the most recent, not-yet-summarized messages.
        return f"### Summary of Past Conversation:\n{self.running_summary}\n\n### Recent Messages:\n{buffer_text}"

# Initialize the SummarizationMemory with a threshold of 4 messages (2 turns).
# This means a summary will be generated after the second full interaction.
summarization_memory = SummarizationMemory(summary_threshold=4)
# Create an AIAgent and inject the summarization memory strategy.
agent = AIAgent(memory_strategy=summarization_memory)

------------------------------------------------------------------------------------------------------------
### 4. Retrieval Based Memory

# Import necessary libraries for numerical operations and similarity search.
import numpy as np
import faiss

# --- Strategy 4: Retrieval-Based Memory ---
# This strategy treats each piece of conversation as a document in a searchable
# database. It uses vector embeddings to find and retrieve the most semantically
# relevant pieces of information from the past to answer a new query. This is the
# core concept behind Retrieval-Augmented Generation (RAG).
class RetrievalMemory(BaseMemoryStrategy):
    def __init__(self, k: int = 2, embedding_dim: int = 3584):
        """
        Initializes the retrieval memory system.
        
        Args:
            k: The number of top relevant documents to retrieve for a given query.
            embedding_dim: The dimension of the vectors generated by the embedding model.
                           For BAAI/bge-multilingual-gemma2, this is 3584.
        """
        # The number of nearest neighbors to retrieve.
        self.k = k
        # The dimensionality of the embedding vectors. Must match the model's output.
        self.embedding_dim = embedding_dim
        # A list to store the original text content of each document.
        self.documents = []
        # Initialize a FAISS index. IndexFlatL2 performs an exhaustive search using
        # L2 (Euclidean) distance, which is effective for a moderate number of vectors.
        self.index = faiss.IndexFlatL2(self.embedding_dim)

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a new conversational turn to the memory. Each part of the turn (user
        input and AI response) is embedded and indexed separately for granular retrieval.
        """
        # We store each part of the turn as a separate document to allow for more
        # precise matching. For example, a query might be similar to a past user
        # statement but not the AI's response in that same turn.
        docs_to_add = [
            f"User said: {user_input}",
            f"AI responded: {ai_response}"
        ]
        for doc in docs_to_add:
            # Generate a numerical vector representation of the document.
            embedding = generate_embedding(doc)
            # Proceed only if the embedding was successfully created.
            if embedding:
                # Store the original text. The index of this document will correspond
                # to the index of its vector in the FAISS index.
                self.documents.append(doc)
                # FAISS requires the input vectors to be a 2D numpy array of float32.
                vector = np.array([embedding], dtype='float32')
                # Add the vector to the FAISS index, making it searchable.
                self.index.add(vector)

    def get_context(self, query: str) -> str:
        """
        Finds the k most relevant documents from memory based on semantic
        similarity to the user's query.
        """
        # If the index has no vectors, there's nothing to search.
        if self.index.ntotal == 0:
            return "No information in memory yet."
        
        # Convert the user's query into an embedding vector.
        query_embedding = generate_embedding(query)
        if not query_embedding:
            return "Could not process query for retrieval."
        
        # Convert the query embedding into the format required by FAISS.
        query_vector = np.array([query_embedding], dtype='float32')
        
        # Perform the search. 'search' returns the distances and the indices
        # of the k nearest neighbors to the query vector.
        distances, indices = self.index.search(query_vector, self.k)
        
        # Use the returned indices to retrieve the original text documents.
        # We check for `i != -1` as FAISS can return -1 for invalid indices.
        retrieved_docs = [self.documents[i] for i in indices[0] if i != -1]
        
        if not retrieved_docs:
            return "Could not find any relevant information in memory."
        
        # Format the retrieved documents into a string to be used as context.
        return "### Relevant Information Retrieved from Memory:\n" + "\n---\n".join(retrieved_docs)

# Initialize the RetrievalMemory with k=2, meaning it will retrieve the top 2 most relevant documents.
retrieval_memory = RetrievalMemory(k=2)
# Create an AIAgent and inject the retrieval memory strategy.
agent = AIAgent(memory_strategy=retrieval_memory)

------------------------------------------------------------------------------------------------------------
### 5. Memory Augmented Transformers
# --- Strategy 5: Memory-Augmented Memory (Simulation) ---
# This strategy simulates the behavior of a Memory-Augmented Transformer model.
# It maintains a short-term sliding window of recent conversation and a separate
# list of "memory tokens" which are important facts extracted from the conversation.
# An LLM call is used to decide if a piece of information is important enough
# to be converted into a persistent memory token.
class MemoryAugmentedMemory(BaseMemoryStrategy):
    def __init__(self, window_size: int = 2):
        """
        Initializes the memory-augmented system.
        
        Args:
            window_size: The number of recent turns to keep in the short-term memory.
        """
        # Use a SlidingWindowMemory instance to manage the recent conversation history.
        self.recent_memory = SlidingWindowMemory(window_size=window_size)
        # A list to store the special, persistent "sticky notes" or key facts.
        self.memory_tokens = []

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds the latest turn to recent memory and then uses an LLM call to decide
        if a new, persistent memory token should be created from this interaction.
        """
        # First, add the new interaction to the short-term sliding window memory.
        self.recent_memory.add_message(user_input, ai_response)
        
        # Construct a prompt for the LLM to analyze the conversation turn and
        # determine if it contains a core fact worth remembering long-term.
        fact_extraction_prompt = (
            f"Analyze the following conversation turn. Does it contain a core fact, preference, or decision that should be remembered long-term? "
            f"Examples include user preferences ('I hate flying'), key decisions ('The budget is $1000'), or important facts ('My user ID is 12345').\n\n"
            f"Conversation Turn:\nUser: {user_input}\nAI: {ai_response}\n\n"
            f"If it contains such a fact, state the fact concisely in one sentence. Otherwise, respond with 'No important fact.'"
        )
        
        # Call the LLM to perform the fact extraction.
        extracted_fact = generate_text("You are a fact-extraction expert.", fact_extraction_prompt)
        
        # Check if the LLM's response indicates that an important fact was found.
        if "no important fact" not in extracted_fact.lower():
            # If a fact was found, print a debug message and add it to our list of memory tokens.
            print(f"--- [Memory Augmentation: New memory token created: '{extracted_fact}'] ---")
            self.memory_tokens.append(extracted_fact)

    def get_context(self, query: str) -> str:
        """
        Constructs the context by combining the short-term recent conversation
        with the list of all long-term, persistent memory tokens.
        """
        # Get the context from the short-term sliding window.
        recent_context = self.recent_memory.get_context(query)
        # Format the list of memory tokens into a readable string.
        memory_token_context = "\n".join([f"- {token}" for token in self.memory_tokens])
        
        # Return the combined context, clearly separating the long-term facts from the recent chat.
        return f"### Key Memory Tokens (Long-Term Facts):\n{memory_token_context}\n\n### Recent Conversation:\n{recent_context}"

# Initialize the MemoryAugmentedMemory with a window size of 2.
# This means the short-term memory will only hold the last two turns.
mem_aug_memory = MemoryAugmentedMemory(window_size=2)
# Create an AIAgent and inject the memory-augmented strategy.
agent = AIAgent(memory_strategy=mem_aug_memory)

------------------------------------------------------------------------------------------------------------
### 6. Hierarchical Optimization for Multi-tasks
# --- Strategy 6: Hierarchical Memory ---
# This strategy combines multiple memory types to create a more sophisticated,
# layered system, mimicking human memory's division into short-term (working)
# and long-term storage.
class HierarchicalMemory(BaseMemoryStrategy):
    def __init__(self, window_size: int = 2, k: int = 2, embedding_dim: int = 3584):
        """
        Initializes the hierarchical memory system.
        
        Args:
            window_size: The size of the short-term working memory (in turns).
            k: The number of documents to retrieve from long-term memory.
            embedding_dim: The dimension of the embedding vectors for long-term memory.
        """
        print("Initializing Hierarchical Memory...")
        # Level 1: Fast, short-term working memory using a sliding window.
        self.working_memory = SlidingWindowMemory(window_size=window_size)
        # Level 2: Slower, durable long-term memory using a retrieval system.
        self.long_term_memory = RetrievalMemory(k=k, embedding_dim=embedding_dim)
        # A simple heuristic: keywords that trigger promotion from working to long-term memory.
        self.promotion_keywords = ["remember", "rule", "preference", "always", "never", "allergic"]

    def add_message(self, user_input: str, ai_response: str):
        """
        Adds a message to working memory and conditionally promotes it to long-term
        memory based on its content.
        """
        # All interactions are added to the fast, short-term working memory.
        self.working_memory.add_message(user_input, ai_response)
        
        # Promotion Logic: Check if the user's input contains a keyword that
        # suggests the information is important and should be stored long-term.
        if any(keyword in user_input.lower() for keyword in self.promotion_keywords):
            print(f"--- [Hierarchical Memory: Promoting message to long-term storage.] ---")
            # If a keyword is found, also add the interaction to the long-term retrieval memory.
            self.long_term_memory.add_message(user_input, ai_response)

    def get_context(self, query: str) -> str:
        """
        Constructs a rich context by combining relevant information from both
        the long-term and short-term memory layers.
        """
        # Retrieve the most recent conversation from the working memory.
        working_context = self.working_memory.get_context(query)
        # Retrieve semantically relevant facts from the long-term memory based on the current query.
        long_term_context = self.long_term_memory.get_context(query)
        
        # Combine both contexts, clearly labeling their sources for the LLM.
        return f"### Retrieved Long-Term Memories:\n{long_term_context}\n\n### Recent Conversation (Working Memory):\n{working_context}"

# Initialize the HierarchicalMemory. It combines a short-term sliding window
# and a long-term retrieval system.
hierarchical_memory = HierarchicalMemory()
# Create an AIAgent and inject the hierarchical memory strategy.
agent = AIAgent(memory_strategy=hierarchical_memory)

------------------------------------------------------------------------------------------------------------
### 7. Graph Based Optimization
# --- Strategy 7: Graph-Based Memory ---
# This strategy represents information as a structured knowledge graph, consisting
# of nodes (entities like 'Sam', 'Innovatech') and edges (relationships like
# 'works_for', 'focuses_on'). It uses the LLM itself to extract these structured
# triples (Subject, Relation, Object) from unstructured conversation text.
class GraphMemory(BaseMemoryStrategy):
    def __init__(self):
        """Initializes the memory with an empty NetworkX directed graph."""
        # A DiGraph is suitable for representing directed relationships (e.g., Sam -> works_for -> Innovatech).
        self.graph = nx.DiGraph()

    def _extract_triples(self, text: str) -> list[tuple[str, str, str]]:
        """
        Uses the LLM to extract knowledge triples (Subject, Relation, Object) from a given text.
        This is a form of "LLM as a Tool" where the model's language understanding is
        used to create structured data.
        """
        print("--- [Graph Memory: Attempting to extract triples from text.] ---")
        # Construct a detailed prompt that instructs the LLM on its role and the desired output format.
        # Providing a clear example is crucial for getting reliable, structured output.
        extraction_prompt = (
            f"You are a knowledge extraction engine. Your task is to extract Subject-Relation-Object triples from the given text. "
            f"Format your output strictly as a list of Python tuples. For example: [('Sam', 'works_for', 'Innovatech'), ('Innovatech', 'focuses_on', 'Energy')]. "
            f"If no triples are found, return an empty list [].\n\n"
            f"Text to analyze:\n\"""{text}\""""
        )
        
        # Call the LLM with the specialized prompt.
        response_text = generate_text("You are an expert knowledge graph extractor.", extraction_prompt)
        
        # Safely parse the string representation of a list of tuples from the LLM's response.
        try:
            # Using regular expressions is a much safer alternative to `eval()`, as it avoids
            # executing arbitrary code that might be maliciously or accidentally included in the LLM's output.
            # This regex looks for patterns matching ('item1', 'item2', 'item3').
            found_triples = re.findall(r"\(['\"](.*?)['\"],\s*['\"](.*?)['\"],\s*['\"](.*?)['\"]\)", response_text)
            print(f"--- [Graph Memory: Extracted triples: {found_triples}] ---")
            return found_triples
        except Exception as e:
            # If parsing fails, log the error and return an empty list to prevent crashes.
            print(f"Could not parse triples from LLM response: {e}")
            return []

    def add_message(self, user_input: str, ai_response: str):
        """Extracts triples from the latest conversation turn and adds them to the knowledge graph."""
        # Combine the user and AI messages to provide full context for extraction.
        full_text = f"User: {user_input}\nAI: {ai_response}"
        # Call the helper method to get structured triples.
        triples = self._extract_triples(full_text)
        # Iterate over the extracted triples.
        for subject, relation, obj in triples:
            # Add an edge to the graph. `add_edge` automatically creates the nodes
            # (subject, obj) if they don't already exist. The relation is stored as an edge attribute.
            # .strip() removes any leading/trailing whitespace for cleaner data.
            self.graph.add_edge(subject.strip(), obj.strip(), relation=relation.strip())

    def get_context(self, query: str) -> str:
        """
        Retrieves context by finding entities from the query in the graph and
        returning all their known relationships.
        """
        # If the graph is empty, there's no context to provide.
        if not self.graph.nodes:
            return "The knowledge graph is empty."
        
        # This is a simple entity linking method: it capitalizes words in the query and checks
        # if they exist as nodes in the graph. A more advanced system would use Natural
        # Language Processing (NLP) to identify named entities more accurately.
        query_entities = [word.capitalize() for word in query.replace('?','').split() if word.capitalize() in self.graph.nodes]
        
        # If no entities from the query are found in our graph, we can't provide specific context.
        if not query_entities:
            return "No relevant entities from your query were found in the knowledge graph."
        
        context_parts = []
        # Use set() to process each unique entity only once.
        for entity in set(query_entities):
            # Find all outgoing edges (e.g., Sam -> works_for -> X)
            for u, v, data in self.graph.out_edges(entity, data=True):
                context_parts.append(f"{u} --[{data['relation']}]--> {v}")
            # Find all incoming edges (e.g., X -> is_located_in -> New York)
            for u, v, data in self.graph.in_edges(entity, data=True):
                context_parts.append(f"{u} --[{data['relation']}]--> {v}")
        
        # Combine the retrieved facts into a single context string, removing duplicates and sorting for consistency.
        return "### Facts Retrieved from Knowledge Graph:\n" + "\n".join(sorted(list(set(context_parts))))

  # Initialize the GraphMemory strategy and the agent.
graph_memory = GraphMemory()
agent = AIAgent(memory_strategy=graph_memory)

# Start the conversation, feeding the agent facts one by one.
agent.chat("A person named Clara works for a company called 'FutureScape'.")
agent.chat("FutureScape is based in Berlin.")
agent.chat("Clara's main project is named 'Odyssey'.")

# Now, ask a question that requires connecting multiple facts.
# The agent needs to link "Clara" to "Odyssey".
agent.chat("Tell me about Clara's project.")

------------------------------------------------------------------------------------------------------------
### 8. Compression & Consolidation Memory
# --- Strategy 8: Compression & Consolidation Memory ---
# This strategy aggressively reduces token usage by using the LLM to compress
# each conversational turn into a single, dense, factual statement.
# This is a more extreme version of summarization, focused purely on
# information density over conversational flow.
class CompressionMemory(BaseMemoryStrategy):
    def __init__(self):
        """Initializes the memory with an empty list to store compressed facts."""
        self.compressed_facts = []

    def add_message(self, user_input: str, ai_response: str):
        """Uses the LLM to compress the latest turn into a concise factual statement."""
        # Combine the user and AI messages into a single text block for compression.
        text_to_compress = f"User: {user_input}\nAI: {ai_response}"
        
        # This prompt is highly specific, instructing the LLM to act as a "data compressor"
        # and to be as concise as possible.
        compression_prompt = (
            f"You are a data compression engine. Your task is to distill the following text into its most essential, factual statement. "
            f"Be as concise as possible, removing all conversational fluff. For example, 'User asked about my name and I, the AI, responded that my name is an AI assistant' should become 'User asked for AI's name.'\n\n"
            f"Text to compress:\n\"{text_to_compress}\""
        )
        
        # Call the LLM with the compression persona.
        compressed_fact = generate_text("You are an expert data compressor.", compression_prompt)
        print(f"--- [Compression Memory: New fact stored: '{compressed_fact}'] ---")
        # Add the highly compressed fact to our memory list.
        self.compressed_facts.append(compressed_fact)

    def get_context(self, query: str) -> str:
        """Returns the list of all compressed facts, formatted as a bulleted list."""
        if not self.compressed_facts:
            return "No compressed facts in memory."
        
        # The context is a simple, clean list of the core facts from the conversation.
        return "### Compressed Factual Memory:\n- " + "\n- ".join(self.compressed_facts)

  # Initialize the CompressionMemory strategy and the agent.
compression_memory = CompressionMemory()
agent = AIAgent(memory_strategy=compression_memory)

# Start the conversation, providing key details one by one.
agent.chat("Okay, I've decided on the venue for the conference. It's going to be the 'Metropolitan Convention Center'.")
agent.chat("The date is confirmed for October 26th, 2025.")
agent.chat("Could you please summarize the key details for the conference plan?")

------------------------------------------------------------------------------------------------------------
### 9. OS-Like Memory Management
# --- Strategy 9: OS-Like Memory Management (Simulation) ---
# This conceptual strategy mimics how a computer's OS manages memory by using
# a small, fast 'active memory' (RAM) and a large, slower 'passive memory' (Disk).
# Information is "paged out" from active to passive memory when space is needed,
# and "paged in" when a query requires old information.
class OSMemory(BaseMemoryStrategy):
    def __init__(self, ram_size: int = 2):
        """
        Initializes the OS-like memory system.

        Args:
            ram_size: The maximum number of conversational turns to keep in active memory (RAM).
        """
        self.ram_size = ram_size
        # The 'RAM' is a deque, holding the most recent turns.
        self.active_memory = deque()
        # The 'Hard Disk' is a dictionary for storing paged-out turns.
        self.passive_memory = {}
        # A counter to give each turn a unique ID.
        self.turn_count = 0

    def add_message(self, user_input: str, ai_response: str):
        """Adds a turn to active memory, paging out the oldest turn to passive memory if RAM is full."""
        turn_id = self.turn_count
        turn_data = f"User: {user_input}\nAI: {ai_response}"
        
        # Check if active memory (RAM) is at capacity.
        if len(self.active_memory) >= self.ram_size:
            # If so, remove the least recently used (oldest) item from active memory.
            lru_turn_id, lru_turn_data = self.active_memory.popleft()
            # Move it to passive memory (the hard disk).
            self.passive_memory[lru_turn_id] = lru_turn_data
            print(f"--- [OS Memory: Paging out Turn {lru_turn_id} to passive storage.] ---")
        
        # Add the new turn to active memory.
        self.active_memory.append((turn_id, turn_data))
        self.turn_count += 1

    def get_context(self, query: str) -> str:
        """Provides RAM context and simulates a 'page fault' to pull from passive memory if needed."""
        # The base context is always what's in the active memory.
        active_context = "\n".join([data for _, data in self.active_memory])
        
        # Simulate a page fault: check if any words in the query match content in passive memory.
        # A real system would use embeddings for this, but keyword search demonstrates the concept.
        paged_in_context = ""
        for turn_id, data in self.passive_memory.items():
            if any(word in data.lower() for word in query.lower().split() if len(word) > 3):
                paged_in_context += f"\n(Paged in from Turn {turn_id}): {data}"
                print(f"--- [OS Memory: Page fault! Paging in Turn {turn_id} from passive storage.] ---")
        
        # Combine the active context with any paged-in context.
        return f"### Active Memory (RAM):\n{active_context}\n\n### Paged-In from Passive Memory (Disk):\n{paged_in_context}"

    def clear(self):
        """Clears both active and passive memory stores."""
        self.active_memory.clear()
        self.passive_memory = {}
        self.turn_count = 0
        print("OS-like memory cleared.")

# Initialize the OS-like memory strategy with a RAM size of 2 turns.
os_memory = OSMemory(ram_size=2)
agent = AIAgent(memory_strategy=os_memory)

# Start the conversation
agent.chat("The secret launch code is 'Orion-Delta-7'.") # This is Turn 0
agent.chat("The weather for the launch looks clear.") # This is Turn 1
agent.chat("The launch window opens at 0400 Zulu.") # This is Turn 2, which pages out Turn 0.

# Now, ask about the paged-out information. This should trigger a page fault.
agent.chat("I need to confirm the launch code.")  
