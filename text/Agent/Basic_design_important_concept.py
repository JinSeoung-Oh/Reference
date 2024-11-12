### From https://medium.com/towards-generative-ai/ai-agents-in-action-interaction-and-workflow-dynamics-f7bd588645a8
"""
1. Introduction to Nature-Inspired AI
   Artificial intelligence (AI) can learn, adapt, and evolve by drawing inspiration from nature’s coordination and cooperation, 
   resembling biological ecosystems. A striking example is an ant colony, where ants use pheromone trails to form efficient paths to resources,
   showcasing natural collective intelligence. Similarly, AI systems could enhance their efficiency and adaptability by using 
   “digital breadcrumbs” — allowing AI agents to share knowledge dynamically, adapting to new situations like human teamwork in complex projects.

2. Key Principles of Collective Intelligence in AI:
   -1. Modularity: A modular structure divides tasks into manageable, specialized roles, inspired by both natural and human collaboration. 
                   For instance, Data-Gathering Agents collect information, Synthesis Agents interpret it, and Decision Agents act on insights. 
                   This framework, exemplified in platforms like CrewAI, organizes AI agents into distinct, autonomous roles, 
                   improving efficiency and precision.

   -2. Design Patterns for Task Handling:
       -a. Pipeline: A step-by-step process where each agent performs a specific task sequentially, ideal for linear workflows.
       -b. Network: Agents function as interconnected nodes, sharing information in real-time to address complex tasks dynamically.
       -c. Forum: Agents communicate in an open-ended, discussion-like format, allowing for brainstorming and adaptability.
       -d. Team: A hybrid structure where agents can work in hierarchy or sequence, with clearly assigned roles, suitable for adaptive, 
                 complex workflows.
3. Memory and AI
   Inspired by human memory’s stages — sensory, short-term, and long-term — AI memory systems can integrate both 
   short- and long-term recall mechanisms. Key components:

   -1. Sensory Memory: Captures brief snapshots of input (like a quick impression) before filtering relevant information for further processing.
   -2. Short-Term Memory: Temporarily holds immediate-use data (e.g., a phone number), allowing quick, task-specific recall.
   -3. Long-Term Memory: Divided into semantic memory (facts, concepts) and episodic memory (personal experiences), 
                         this structure organizes information for extensive recall.

4. Blueprint for Human-Like AI Memory:
   -1. Dynamic Memory Retrieval: Context-sensitive memory recall allows the AI to access relevant information from prior sessions, 
                                 providing continuity and informed responses.
   -2. Adaptive Learning: The AI updates its long-term memory with insights from interactions, enhancing adaptability over time.
   -3. Entity-Based Organization: AI memory structures can recognize entities, organizing information by topics and metadata, 
                                  similar to how human memory categorizes related knowledge.
"""
---------------------------------------------------
# Author: Benjamin Chu
# Description: Custom memory tool for storing and indexing conversation entries for AI agents.
# Created: 2024-11-02
# Purpose: Supports both short-term (session-based) and long-term (vector-based) memory structures with topic and entity recognition.

@tool
def save_to_memory(collection, user_id: str, messages: List) -> Dict[str, str]:
    all_data = []
    success = True
    
    for message in messages:
        role = "user" if "HumanMessage" in str(type(message)) else "assistant"
        entry_id = message.id
        content = message.content
        
        entry = {
            "user_id": user_id,
            "process_id": entry_id,
            "role": role,
            "text": None,
            "topic": None,
            "embeddings": None
        }

        # Format and save all assistant messages as conversation entries
        if role == "assistant" and "answer" in content:
            try:
                response_data = eval(content.strip("```"))
                entry["text"] = response_data.get("answer")
                entry["topic"] = response_data.get("topic")
                
                if entry["text"]:
                    embedding = list(embedding_model.embed(entry["text"]))
                    entry["embeddings"] = embedding[0]
                
                all_data.append(entry)
            except (SyntaxError, ValueError) as e:
                logger.error(f"Failed to parse assistant message content: {e}")
                success = False
        elif role == "user":
            entry["text"] = content
    
    try:
        collection.insert(all_data)
    except Exception as e:
        logger.error(f"Failed to insert data into the collection: {e}")
        success = False

    # Release or drop the existing collection index
    collection.release()
    collection.drop_index(index_name='memory_idx')

    # Create an index for the embedding field
    collection.create_index(
            field_name="embeddings",
            index_params={
                "index_type": "HNSW",
                "metric_type": "COSINE",
                "params": {"M": 64, "efConstruction": 512},
            },
            index_name="memory_idx"
    )

    # Reload collection
    collection.load()
    
    return {
        "status": "Success" if success else "Fail",
        "message": "Data loaded and indexed successfully" if success else "Data loading failed"
    }
---------------------------------------------------
"""
In the CrewAI framework, enabling memory functions activates several types of memory — Short-Term, Long-Term, Entity, 
and Contextual Memory — to improve the AI agent’s coherence and recall abilities. 
However, there’s currently no simple way to inspect these memory interactions directly, requiring users to add logging manually,
which detracts from the framework’s seamless memory experience.

1. Short-Term Memory with LangGraph Checkpointer:
   For session-based memory (thread-specific), LangGraph’s checkpointer is used. 
   This mechanism enables the agent to recall relevant information within a conversation thread, such as an email chain,
   maintaining continuity and context within a session.

2. Long-Term Memory Challenges:
   -1. InMemoryStore Limitations
       The JSON-based InMemoryStore in LangGraph stores data under specific namespaces and keys, but it lacks semantic similarity search. 
       This limitation complicates retrieval of contextually relevant information, 
       which is essential for nuanced interactions and approximate matches.
   -2. Graph-Based Memory Considerations
       Initially, a graph-based approach for long-term memory was considered, 
       as graphs excel at capturing semantic relationships and enabling proximity-based filtering.
       This approach could use time-travel traversal, where memory retrieval would start from the most relevant node based on similarity,
       exploring connected neighbors. 
       However, managing high-dimensional similarity queries within a graph setup could be complex and resource-intensive.
   -3. AriGraph Example
       AriGraph exemplifies an advanced system combining knowledge graphs with semantic and episodic memory,
       drawing from neuroscience to support continual learning, reasoning, and planning.
       Despite its effectiveness for complex memory systems, graph scalability remains a challenge in high-dimensional contexts.

3. Milvus for Vector-Based Memory Storage:
   -1. Solution: Milvus, a high-performance vector database, was chosen for long-term memory management due to its built-in support 
                 for embeddings and approximate nearest neighbor (ANN) search, providing efficient similarity-based querying.
   -2. Advantages: Milvus simplifies the complexity of memory management by enabling direct, scalable queries based on vector similarity. 
                   This ensures that contextually similar memory entries are retrieved quickly and effectively, 
                   avoiding the overhead of graph traversal while maintaining robust performance.
"""
---------------------------------------------------
# Example current query for memory search based on agentic workflow discussions
query = "how do agents work together to collaborate?"
query_embedding = embedding_model.embed(query)[0]

...

# Configure search parameters for vector similarity retrieval
search_params = {
    "metric_type": "COSINE",
    "params": {"M": 64, "efConstruction": 512}
}

# Perform a vector search on the collection to find relevant memories
results = collection.search(
    data=[query_embedding],
    anns_field="embeddings",
    param=search_params,
    limit=3,
    expr="topic == 'agentic_workflow'",
    output_fields=['text'],
    consistency_level="Strong"
)
---------------------------------------------------
"""
This setup focuses on enhancing agent collaboration within workflows, enabling AI agents to recall relevant context and make informed decisions 
based on prior interactions.

-1. Query Embedding and Vector-Based Search:
    When a query, such as “how do agents work together to collaborate?”, is submitted, it’s embedded using FastEmbeddings
    to capture semantic nuances.
    This dense vector is used with Milvus, a vector database optimized for high-performance similarity search. 
    Milvus identifies the top three most contextually similar memories through cosine similarity.

-2. Topic-Based Filtering:
    A filter narrows the search results to memories tagged with “agentic_workflow,” focusing specifically on information related
    to agent collaboration in workflows.
    Llama 3.2, the language model, determines this topic during initial query processing, 
    refining the search to retrieve memories pertinent to agent cooperation.

-3. Memory Context Integration:
    Retrieved memories containing only the ‘text’ field are provided as context to the agent. 
    This allows the agent to leverage relevant past discussions on “agentic workflows,” enhancing coherence and depth in its responses.
    This approach empowers the agent to handle complex workflows by grounding responses in previously stored knowledge.

-4. User Interface and Multi-Turn Interaction:
    A dedicated user interface illustrates the agent’s memory-aware response flow. 
    Users interact across multiple turns, with the agent maintaining contextual awareness, reflecting past exchanges, 
    and responding dynamically to evolving conversation threads.

-5. Dual Memory System Benefits:
    -a. Short-Term Memory: Manages context within the current session, ideal for resolving references in ongoing conversations,
                           ensuring accuracy and relevance.
    -b. Long-Term Memory: Supports continuity by storing and retrieving contextually similar memories, aiding in multi-turn interaction continuity.

In conclusion, this memory architecture — combining vector similarity search, topic filtering, 
and session-specific memory — allows agents to deliver contextually rich and coherent responses, supporting complex workflows. 
The next steps will cover building workflows with function calls to integrate APIs and custom tools,
alongside practical blueprints for implementation.
"""

