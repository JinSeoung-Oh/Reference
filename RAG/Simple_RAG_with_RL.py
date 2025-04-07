### From https://levelup.gitconnected.com/maximizing-simple-rag-performance-using-rl-in-python-d4c14cbadf59

!git clone https://github.com/FareedKhan-dev/rag-with-rl.git
!pip install -r rag-with-rl/requirements.txt

import os
from openai import OpenAI
import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Union

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",  # Base URL for (eg. ollama api, anyother llm api provider)
    api_key= os.environ["OPENAI_API_KEY"]  # API key for authentication 
)

def load_documents(directory_path: str) -> List[str]:
    """
    Load all text documents from the specified directory.

    Args:
        directory_path (str): Path to the directory containing text files.

    Returns:
        List[str]: A list of strings, where each string is the content of a text file.
    """
    documents = []  # Initialize an empty list to store document contents
    for filename in os.listdir(directory_path):  # Iterate through all files in the directory
        if filename.endswith(".txt"):  # Check if the file has a .txt extension
            # Open the file in read mode with UTF-8 encoding and append its content to the list
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                documents.append(file.read())
    return documents  # Return the list of document contents

def split_into_chunks(documents: List[str], chunk_size: int = 30) -> List[str]:
    """
    Split documents into smaller chunks of specified size.

    Args:
        documents (List[str]): A list of document strings to be split into chunks.
        chunk_size (int): The maximum number of words in each chunk. Default is 100.

    Returns:
        List[str]: A list of chunks, where each chunk is a string containing up to `chunk_size` words.
    """
    chunks = []  # Initialize an empty list to store the chunks
    for doc in documents:  # Iterate through each document
        words = doc.split()  # Split the document into words
        # Create chunks of the specified size
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])  # Join words to form a chunk
            chunks.append(chunk)  # Add the chunk to the list
    return chunks  # Return the list of chunks

def preprocess_text(text: str) -> str:
    """
    Preprocess the input text by converting it to lowercase and removing special characters.

    Args:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text with only alphanumeric characters and spaces.
    """
    # Convert the text to lowercase
    text = text.lower()
    # Remove special characters, keeping only alphanumeric characters and spaces
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

def preprocess_chunks(chunks: List[str]) -> List[str]:
    """
    Apply preprocessing to all text chunks.

    Args:
        chunks (List[str]): A list of text chunks to preprocess.

    Returns:
        List[str]: A list of preprocessed text chunks.
    """
    # Apply the preprocess_text function to each chunk in the list
    return [preprocess_text(chunk) for chunk in chunks]

def generate_embeddings_batch(chunks_batch: List[str], model: str = "BAAI/bge-en-icl") -> List[List[float]]:
    """
    Generate embeddings for a batch of text chunks using the OpenAI client.

    Args:
        chunks_batch (List[str]): A batch of text chunks to generate embeddings for.
        model (str): The model to use for embedding generation. Default is "BAAI/bge-en-icl".

    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.
    """
    # Use the OpenAI client to create embeddings for the input batch
    response = client.embeddings.create(
        model=model,  # Specify the model to use for embedding generation
        input=chunks_batch  # Provide the batch of text chunks as input
    )
    # Extract embeddings from the response and return them
    embeddings = [item.embedding for item in response.data]
    return embeddings

def generate_embeddings(chunks: List[str], batch_size: int = 10) -> np.ndarray:
    """
    Generate embeddings for all text chunks in batches.

    Args:
        chunks (List[str]): A list of text chunks to generate embeddings for.
        batch_size (int): The number of chunks to process in each batch. Default is 10.

    Returns:
        np.ndarray: A NumPy array containing embeddings for all chunks.
    """
    all_embeddings = []  # Initialize an empty list to store all embeddings

    # Iterate through the chunks in batches
    for i in range(0, len(chunks), batch_size):
        # Extract the current batch of chunks
        batch = chunks[i:i + batch_size]
        # Generate embeddings for the current batch
        embeddings = generate_embeddings_batch(batch)
        # Extend the list of all embeddings with the embeddings from the current batch
        all_embeddings.extend(embeddings)

    # Convert the list of embeddings to a NumPy array and return it
    return np.array(all_embeddings)

def save_embeddings(embeddings: np.ndarray, output_file: str) -> None:
    """
    Save embeddings to a JSON file.

    Args:
        embeddings (np.ndarray): A NumPy array containing the embeddings to save.
        output_file (str): The path to the output JSON file where embeddings will be saved.

    Returns:
        None
    """
    # Open the specified file in write mode with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as file:
        # Convert the NumPy array to a list and save it as JSON
        json.dump(embeddings.tolist(), file)

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Compute the cosine similarity between two vectors.

    Args:
        vec1 (np.ndarray): The first vector.
        vec2 (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors, ranging from -1 to 1.
    """
    # Compute the dot product of the two vectors
    dot_product = np.dot(vec1, vec2)
    # Compute the magnitude (norm) of the first vector
    norm_vec1 = np.linalg.norm(vec1)
    # Compute the magnitude (norm) of the second vector
    norm_vec2 = np.linalg.norm(vec2)
    # Return the cosine similarity as the ratio of the dot product to the product of the norms
    return dot_product / (norm_vec1 * norm_vec2)

def add_to_vector_store(embeddings: np.ndarray, chunks: List[str]) -> None:
    """
    Add embeddings and their corresponding text chunks to the vector store.

    Args:
        embeddings (np.ndarray): A NumPy array containing the embeddings to add.
        chunks (List[str]): A list of text chunks corresponding to the embeddings.

    Returns:
        None
    """
    # Iterate over embeddings and chunks simultaneously
    for embedding, chunk in zip(embeddings, chunks):
        # Add each embedding and its corresponding chunk to the vector store
        # Use the current length of the vector store as the unique key
        vector_store[len(vector_store)] = {"embedding": embedding, "chunk": chunk}

def similarity_search(query_embedding: np.ndarray, top_k: int = 5) -> List[str]:
    """
    Perform similarity search in the vector store and return the top_k most similar chunks.

    Args:
        query_embedding (np.ndarray): The embedding vector of the query.
        top_k (int): The number of most similar chunks to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top_k most similar text chunks.
    """
    similarities = []  # Initialize a list to store similarity scores and corresponding keys

    # Iterate through all items in the vector store
    for key, value in vector_store.items():
        # Compute the cosine similarity between the query embedding and the stored embedding
        similarity = cosine_similarity(query_embedding, value["embedding"])
        # Append the key and similarity score as a tuple to the list
        similarities.append((key, similarity))

    # Sort the list of similarities in descending order based on the similarity score
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    # Retrieve the top_k most similar chunks based on their keys
    return [vector_store[key]["chunk"] for key, _ in similarities[:top_k]]

def retrieve_relevant_chunks(query_text: str, top_k: int = 5) -> List[str]:
    """
    Retrieve the most relevant document chunks for a given query text.

    Args:
        query_text (str): The query text for which relevant chunks are to be retrieved.
        top_k (int): The number of most relevant chunks to retrieve. Default is 5.

    Returns:
        List[str]: A list of the top_k most relevant text chunks.
    """
    # Generate embedding for the query text using the embedding model
    query_embedding = generate_embeddings([query_text])[0]
    
    # Perform similarity search to find the most relevant chunks
    relevant_chunks = similarity_search(query_embedding, top_k=top_k)
    
    # Return the list of relevant chunks
    return relevant_chunks

def construct_prompt(query: str, context_chunks: List[str]) -> str:
    """
    Construct a prompt by combining the query with the retrieved context chunks.

    Args:
        query (str): The query text for which the prompt is being constructed.
        context_chunks (List[str]): A list of relevant context chunks to include in the prompt.

    Returns:
        str: The constructed prompt to be used as input for the LLM.
    """
    # Combine all context chunks into a single string, separated by newlines
    context = "\n".join(context_chunks)
    
    # Define the system message to guide the LLM's behavior
    system_message = (
        "You are a helpful assistant. Only use the provided context to answer the question. "
        "If the context doesn't contain the information needed, say 'I don't have enough information to answer this question.'"
    )
    
    # Construct the final prompt by combining the system message, context, and query
    prompt = f"System: {system_message}\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:"
    
    return prompt

def generate_response(
    prompt: str,
    model: str = "google/gemma-2-2b-it",
    max_tokens: int = 512,
    temperature: float = 1,
    top_p: float = 0.9,
    top_k: int = 50
) -> str:
    """
    Generate a response from the OpenAI chat model based on the constructed prompt.

    Args:
        prompt (str): The input prompt to provide to the chat model.
        model (str): The model to use for generating the response. Default is "google/gemma-2-2b-it".
        max_tokens (int): Maximum number of tokens in the response. Default is 512.
        temperature (float): Sampling temperature for response diversity. Default is 0.5.
        top_p (float): Probability mass for nucleus sampling. Default is 0.9.
        top_k (int): Number of highest probability tokens to consider. Default is 50.

    Returns:
        str: The generated response from the chat model.
    """
    # Use the OpenAI client to create a chat completion
    response = client.chat.completions.create(
        model=model,  # Specify the model to use for generating the response
        max_tokens=max_tokens,  # Maximum number of tokens in the response
        temperature=temperature,  # Sampling temperature for response diversity
        top_p=top_p,  # Probability mass for nucleus sampling
        extra_body={  # Additional parameters for the request
            "top_k": top_k  # Number of highest probability tokens to consider
        },
        messages=[  # List of messages to provide context for the chat model
            {
                "role": "user",  # Role of the message sender (user in this case)
                "content": [  # Content of the message
                    {
                        "type": "text",  # Type of content (text in this case)
                        "text": prompt  # The actual prompt text
                    }
                ]
            }
        ]
    )
    # Return the content of the first choice in the response
    return response.choices[0].message.content

def basic_rag_pipeline(query: str) -> str:
    """
    Implement the basic Retrieval-Augmented Generation (RAG) pipeline:
    retrieve relevant chunks, construct a prompt, and generate a response.

    Args:
        query (str): The input query for which a response is to be generated.

    Returns:
        str: The generated response from the LLM based on the query and retrieved context.
    """
    # Step 1: Retrieve the most relevant chunks for the given query
    relevant_chunks: List[str] = retrieve_relevant_chunks(query)
    
    # Step 2: Construct a prompt using the query and the retrieved chunks
    prompt: str = construct_prompt(query, relevant_chunks)
    
    # Step 3: Generate a response from the LLM using the constructed prompt
    response: str = generate_response(prompt)
    
    # Return the generated response
    return response

def define_state(
    query: str, 
    context_chunks: List[str], 
    rewritten_query: str = None, 
    previous_responses: List[str] = None, 
    previous_rewards: List[float] = None
) -> dict:
    """
    Define the state representation for the reinforcement learning agent.
    
    Args:
        query (str): The original user query.
        context_chunks (List[str]): Retrieved context chunks from the knowledge base.
        rewritten_query (str, optional): A reformulated version of the original query.
        previous_responses (List[str], optional): List of previously generated responses.
        previous_rewards (List[float], optional): List of rewards received for previous actions.
    
    Returns:
        dict: A dictionary representing the current state with all relevant information.
    """
    state = {
        "original_query": query,                                    # The initial query from the user
        "current_query": rewritten_query if rewritten_query else query,  # Current version of the query (may be rewritten)
        "context": context_chunks,                                 # Retrieved context chunks from the knowledge base
        "previous_responses": previous_responses if previous_responses else [],  # History of generated responses
        "previous_rewards": previous_rewards if previous_rewards else []         # History of received rewards
    }
    return state

def define_action_space() -> List[str]:
    """
    Define the set of possible actions the reinforcement learning agent can take.
    
    Actions include:
    - rewrite_query: Reformulate the original query to improve retrieval
    - expand_context: Retrieve additional context chunks
    - filter_context: Remove irrelevant context chunks
    - generate_response: Generate a response based on current query and context
    
    Returns:
        List[str]: A list of available actions.
    """

    # Define the set of actions the agent can take
    actions = ["rewrite_query", "expand_context", "filter_context", "generate_response"]
    return actions

def calculate_reward(response: str, ground_truth: str) -> float:
    """
    Calculate a reward value by comparing the generated response to the ground truth.
    
    Uses cosine similarity between the embeddings of the response and ground truth
    to determine how close the response is to the expected answer.
    
    Args:
        response (str): The generated response from the RAG pipeline.
        ground_truth (str): The expected correct answer.
    
    Returns:
        float: A reward value between -1 and 1, where higher values indicate 
               greater similarity to the ground truth.
    """
    # Generate embeddings for both the response and ground truth
    response_embedding = generate_embeddings([response])[0]
    ground_truth_embedding = generate_embeddings([ground_truth])[0]
    
    # Calculate cosine similarity between the embeddings as the reward
    similarity = cosine_similarity(response_embedding, ground_truth_embedding)
    return similarity

def rewrite_query(
    query: str, 
    context_chunks: List[str], 
    model: str = "google/gemma-2-2b-it", 
    max_tokens: int = 100, 
    temperature: float = 0.3
) -> str:
    """
    Use the LLM to rewrite the query for better document retrieval.

    Args:
        query (str): The original query text.
        context_chunks (List[str]): A list of context chunks retrieved so far.
        model (str): The model to use for generating the rewritten query. Default is "google/gemma-2-2b-it".
        max_tokens (int): Maximum number of tokens in the rewritten query. Default is 100.
        temperature (float): Sampling temperature for response diversity. Default is 0.3.

    Returns:
        str: The rewritten query optimized for document retrieval.
    """
    # Construct a prompt for the LLM to rewrite the query
    rewrite_prompt = f"""
    You are a query optimization assistant. Your task is to rewrite the given query to make it more effective 
    for retrieving relevant information. The query will be used for document retrieval.
    
    Original query: {query}
    
    Based on the context retrieved so far:
    {' '.join(context_chunks[:2]) if context_chunks else 'No context available yet'}
    
    Rewrite the query to be more specific and targeted to retrieve better information.
    Rewritten query:
    """
    
    # Use the LLM to generate a rewritten query
    response = client.chat.completions.create(
        model=model, # Specify the model to use for generating the response
        max_tokens=max_tokens, # Maximum number of tokens in the response
        temperature=temperature, # Sampling temperature for response diversity
        messages=[
            {
                "role": "user",
                "content": rewrite_prompt
            }
        ]
    )
    
    # Extract and return the rewritten query from the response
    rewritten_query = response.choices[0].message.content.strip()
    return rewritten_query

def expand_context(query: str, current_chunks: List[str], top_k: int = 3) -> List[str]:
    """
    Expand the context by retrieving additional chunks.

    Args:
        query (str): The query text for which additional context is needed.
        current_chunks (List[str]): The current list of context chunks.
        top_k (int): The number of additional chunks to retrieve. Default is 3.

    Returns:
        List[str]: The expanded list of context chunks including new unique chunks.
    """
    # Retrieve more chunks than currently available
    additional_chunks = retrieve_relevant_chunks(query, top_k=top_k + len(current_chunks))
    
    # Filter out chunks that are already in the current context
    new_chunks = []
    for chunk in additional_chunks:
        if chunk not in current_chunks:
            new_chunks.append(chunk)
    
    # Add new unique chunks to the current context, limited to top_k
    expanded_context = current_chunks + new_chunks[:top_k]
    return expanded_context

def filter_context(query: str, context_chunks: List[str]) -> List[str]:
    """
    Filter the context to keep only the most relevant chunks.

    Args:
        query (str): The query text for which relevance is calculated.
        context_chunks (List[str]): The list of context chunks to filter.

    Returns:
        List[str]: A filtered list of the most relevant context chunks.
    """
    if not context_chunks:
        return []
        
    # Generate embeddings for the query and each chunk
    query_embedding = generate_embeddings([query])[0]
    chunk_embeddings = [generate_embeddings([chunk])[0] for chunk in context_chunks]
    
    # Calculate relevance scores for each chunk
    relevance_scores = []
    for chunk_embedding in chunk_embeddings:
        score = cosine_similarity(query_embedding, chunk_embedding)
        relevance_scores.append(score)
    
    # Sort chunks by relevance scores in descending order
    sorted_chunks = [x for _, x in sorted(zip(relevance_scores, context_chunks), reverse=True)]
    
    # Keep the top 5 most relevant chunks or fewer if less than 5 are available
    filtered_chunks = sorted_chunks[:min(5, len(sorted_chunks))]
    
    return filtered_chunks

def policy_network(
    state: dict, 
    action_space: List[str], 
    epsilon: float = 0.2
) -> str:
    """
    Define a policy network to select an action based on the current state using an epsilon-greedy strategy.

    Args:
        state (dict): The current state of the environment, including query, context, responses, and rewards.
        action_space (List[str]): The list of possible actions the agent can take.
        epsilon (float): The probability of choosing a random action for exploration. Default is 0.2.

    Returns:
        str: The selected action from the action space.
    """
    # Use epsilon-greedy strategy: random exploration vs. exploitation
    if np.random.random() < epsilon:
        # Exploration: randomly select an action from the action space
        action = np.random.choice(action_space)
    else:
        # Exploitation: select the best action based on the current state using a simple heuristic

        # If there are no previous responses, prioritize rewriting the query
        if len(state["previous_responses"]) == 0:
            action = "rewrite_query"
        # If there are previous responses but the rewards are low, try expanding the context
        elif state["previous_rewards"] and max(state["previous_rewards"]) < 0.7:
            action = "expand_context"
        # If the context has too many chunks, try filtering the context
        elif len(state["context"]) > 5:
            action = "filter_context"
        # Otherwise, generate a response
        else:
            action = "generate_response"
    
    return action

def rl_step(
    state: dict, 
    action_space: List[str], 
    ground_truth: str
) -> tuple[dict, str, float, str]:
    """
    Perform a single RL step: select an action, execute it, and calculate the reward.

    Args:
        state (dict): The current state of the environment, including query, context, responses, and rewards.
        action_space (List[str]): The list of possible actions the agent can take.
        ground_truth (str): The expected correct answer to calculate the reward.

    Returns:
        tuple: A tuple containing:
            - state (dict): The updated state after executing the action.
            - action (str): The action selected by the policy network.
            - reward (float): The reward received for the action.
            - response (str): The response generated (if applicable).
    """
    # Select an action using the policy network
    action: str = policy_network(state, action_space)
    response: str = None  # Initialize response as None
    reward: float = 0  # Initialize reward as 0

    # Execute the selected action
    if action == "rewrite_query":
        # Rewrite the query to improve retrieval
        rewritten_query: str = rewrite_query(state["original_query"], state["context"])
        state["current_query"] = rewritten_query  # Update the current query in the state
        # Retrieve new context based on the rewritten query
        new_context: List[str] = retrieve_relevant_chunks(rewritten_query)
        state["context"] = new_context  # Update the context in the state

    elif action == "expand_context":
        # Expand the context by retrieving additional chunks
        expanded_context: List[str] = expand_context(state["current_query"], state["context"])
        state["context"] = expanded_context  # Update the context in the state

    elif action == "filter_context":
        # Filter the context to keep only the most relevant chunks
        filtered_context: List[str] = filter_context(state["current_query"], state["context"])
        state["context"] = filtered_context  # Update the context in the state

    elif action == "generate_response":
        # Construct a prompt using the current query and context
        prompt: str = construct_prompt(state["current_query"], state["context"])
        # Generate a response using the LLM
        response: str = generate_response(prompt)
        # Calculate the reward based on the similarity between the response and the ground truth
        reward: float = calculate_reward(response, ground_truth)
        # Update the state with the new response and reward
        state["previous_responses"].append(response)
        state["previous_rewards"].append(reward)

    # Return the updated state, selected action, reward, and response
    return state, action, reward, response

def initialize_training_params() -> Dict[str, Union[float, int]]:
    """
    Initialize training parameters such as learning rate, number of episodes, and discount factor.

    Returns:
        Dict[str, Union[float, int]]: A dictionary containing the initialized training parameters.
    """
    params = {
        "learning_rate": 0.01,  # Learning rate for policy updates
        "num_episodes": 100,   # Total number of training episodes
        "discount_factor": 0.99  # Discount factor for future rewards
    }
    return params

def update_policy(
    policy: Dict[str, Dict[str, Union[float, str]]], 
    state: Dict[str, object], 
    action: str, 
    reward: float, 
    learning_rate: float
) -> Dict[str, Dict[str, Union[float, str]]]:
    """
    Update the policy based on the reward received.

    Args:
        policy (Dict[str, Dict[str, Union[float, str]]]): The current policy to be updated.
        state (Dict[str, object]): The current state of the environment.
        action (str): The action taken by the agent.
        reward (float): The reward received for the action.
        learning_rate (float): The learning rate for updating the policy.

    Returns:
        Dict[str, Dict[str, Union[float, str]]]: The updated policy.
    """
    # Example: Simple policy update (to be replaced with a proper RL algorithm)
    policy[state["query"]] = {
        "action": action,  # Store the action taken
        "reward": reward   # Store the reward received
    }
    return policy

def track_progress(
    episode: int, 
    reward: float, 
    rewards_history: List[float]
) -> List[float]:
    """
    Track the training progress by storing rewards for each episode.

    Args:
        episode (int): The current episode number.
        reward (float): The reward received in the current episode.
        rewards_history (List[float]): A list to store the rewards for all episodes.

    Returns:
        List[float]: The updated rewards history.
    """
    # Append the current reward to the rewards history
    rewards_history.append(reward)
    
    # Print progress every 10 episodes
    print(f"Episode {episode}: Reward = {reward}")
    
    return rewards_history

def training_loop(
    query_text: str, 
    ground_truth: str, 
    params: Optional[Dict[str, Union[float, int]]] = None
) -> Tuple[Dict[str, Dict[str, Union[float, str]]], List[float], List[List[str]], Optional[str]]:
    """
    Implement the training loop for RL-enhanced RAG.

    Args:
        query_text (str): The input query text for the RAG pipeline.
        ground_truth (str): The expected correct answer for the query.
        params (Optional[Dict[str, Union[float, int]]]): Training parameters such as learning rate, 
            number of episodes, and discount factor. If None, default parameters are initialized.

    Returns:
        Tuple: A tuple containing:
            - policy (Dict[str, Dict[str, Union[float, str]]]): The updated policy after training.
            - rewards_history (List[float]): A list of rewards received in each episode.
            - actions_history (List[List[str]]): A list of actions taken in each episode.
            - best_response (Optional[str]): The best response generated during training.
    """
    # Initialize training parameters if not provided
    if params is None:
        params = initialize_training_params()
    
    # Initialize variables to track progress
    rewards_history: List[float] = []  # List to store rewards for each episode
    actions_history: List[List[str]] = []  # List to store actions taken in each episode
    policy: Dict[str, Dict[str, Union[float, str]]] = {}  # Policy dictionary to store actions and rewards
    action_space: List[str] = define_action_space()  # Define the action space
    best_response: Optional[str] = None  # Variable to store the best response
    best_reward: float = -1  # Initialize the best reward to a very low value
    
    # Get initial performance from the simple RAG pipeline for comparison
    simple_response: str = basic_rag_pipeline(query_text)
    simple_reward: float = calculate_reward(simple_response, ground_truth)
    print(f"Simple RAG reward: {simple_reward:.4f}")

    # Start the training loop
    for episode in range(params["num_episodes"]):
        # Reset the environment with the same query
        context_chunks: List[str] = retrieve_relevant_chunks(query_text)
        state: Dict[str, object] = define_state(query_text, context_chunks)
        episode_reward: float = 0  # Initialize the reward for the current episode
        episode_actions: List[str] = []  # Initialize the list of actions for the current episode
        
        # Maximum number of steps per episode to prevent infinite loops
        for step in range(10):
            # Perform a single RL step
            state, action, reward, response = rl_step(state, action_space, ground_truth)
            episode_actions.append(action)  # Record the action taken
            
            # If a response is generated, end the episode
            if response:
                episode_reward = reward  # Update the episode reward
                
                # Track the best response and reward
                if reward > best_reward:
                    best_reward = reward
                    best_response = response
                
                break  # Exit the loop as the episode ends
        
        # Update rewards and actions history
        rewards_history.append(episode_reward)
        actions_history.append(episode_actions)
        
        # Print progress every 5 episodes
        if episode % 5 == 0:
            print(f"Episode {episode}: Reward = {episode_reward:.4f}, Actions = {episode_actions}")
    
    # Compare the best RL-enhanced RAG reward with the simple RAG reward
    improvement: float = best_reward - simple_reward
    print(f"\nTraining completed:")
    print(f"Simple RAG reward: {simple_reward:.4f}")
    print(f"Best RL-enhanced RAG reward: {best_reward:.4f}")
    print(f"Improvement: {improvement:.4f} ({improvement * 100:.2f}%)")

    return policy, rewards_history, actions_history, best_response

def compare_rag_approaches(query_text: str, ground_truth: str) -> Tuple[str, str, float, float]:
    """
    Compare the outputs of simple RAG versus RL-enhanced RAG.

    Args:
        query_text (str): The input query text for the RAG pipeline.
        ground_truth (str): The expected correct answer for the query.

    Returns:
        Tuple[str, str, float, float]: A tuple containing:
            - simple_response (str): The response generated by the simple RAG pipeline.
            - best_rl_response (str): The best response generated by the RL-enhanced RAG pipeline.
            - simple_similarity (float): The similarity score of the simple RAG response to the ground truth.
            - rl_similarity (float): The similarity score of the RL-enhanced RAG response to the ground truth.
    """
    print("=" * 80)
    print(f"Query: {query_text}")
    print("=" * 80)
    
    # Step 1: Generate a response using the simple RAG pipeline
    # The basic RAG pipeline retrieves relevant chunks and generates a response without reinforcement learning.
    simple_response: str = basic_rag_pipeline(query_text)
    # Calculate the similarity score between the simple RAG response and the ground truth.
    simple_similarity: float = calculate_reward(simple_response, ground_truth)
    
    print("\nSimple RAG Output:")
    print("-" * 40)
    print(simple_response)
    print(f"Similarity to ground truth: {simple_similarity:.4f}")
    
    # Step 2: Train the RL-enhanced RAG model
    print("\nTraining RL-enhanced RAG model...")
    # Initialize training parameters (e.g., learning rate, number of episodes, discount factor).
    params: Dict[str, float | int] = initialize_training_params()
    # Set the number of episodes to a smaller value for demonstration purposes.
    params["num_episodes"] = 5
    
    # Run the training loop for the RL-enhanced RAG model.
    # This loop trains the model to optimize its responses using reinforcement learning.
    _, rewards_history, actions_history, best_rl_response = training_loop(
        query_text, ground_truth, params
    )
    
    # If no response was generated during training, generate one using the current query and context.
    if best_rl_response is None:
        # Retrieve relevant chunks for the query.
        context_chunks: List[str] = retrieve_relevant_chunks(query_text)
        # Construct a prompt using the query and retrieved context.
        prompt: str = construct_prompt(query_text, context_chunks)
        # Generate a response using the language model.
        best_rl_response: str = generate_response(prompt)
    
    # Calculate the similarity score between the RL-enhanced RAG response and the ground truth.
    rl_similarity: float = calculate_reward(best_rl_response, ground_truth)
    
    print("\nRL-enhanced RAG Output:")
    print("-" * 40)
    print(best_rl_response)
    print(f"Similarity to ground truth: {rl_similarity:.4f}")
    
    # Step 3: Evaluate and compare the results
    # Calculate the improvement in similarity score achieved by the RL-enhanced RAG model.
    improvement: float = rl_similarity - simple_similarity
    
    print("\nEvaluation Results:")
    print("-" * 40)
    print(f"Simple RAG similarity to ground truth: {simple_similarity:.4f}")
    print(f"RL-enhanced RAG similarity to ground truth: {rl_similarity:.4f}")
    print(f"Improvement: {improvement * 100:.2f}%")
    
    # Step 4: Plot the reward history (if there are enough episodes and matplotlib is available)
    if len(rewards_history) > 1:
        try:
            import matplotlib.pyplot as plt
            # Create a plot to visualize the reward history during RL training.
            plt.figure(figsize=(10, 6))
            plt.plot(rewards_history)
            plt.title('Reward History During RL Training')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.show()
        except ImportError:
            # If matplotlib is not available, print a message instead of plotting.
            print("Matplotlib not available for plotting rewards")
    
    # Return the results: responses and similarity scores for both approaches.
    return simple_response, best_rl_response, simple_similarity, rl_similarity

# Specify the directory path containing the text files
directory_path = "data"

# Load all text documents from the specified directory
documents = load_documents(directory_path)

# Split the loaded documents into smaller chunks of text
chunks = split_into_chunks(documents)

# Preprocess the chunks (e.g., lowercasing, removing special characters)
preprocessed_chunks = preprocess_chunks(chunks)

# Ensure the chunks are preprocessed before generating embeddings
preprocessed_chunks = preprocess_chunks(chunks)

# Generate embeddings for the preprocessed chunks
embeddings = generate_embeddings(preprocessed_chunks)

# Save the generated embeddings to a JSON file named "embeddings.json"
save_embeddings(embeddings, "embeddings.json")

vector_store: dict[int, dict[str, object]] = {}

# Add the generated embeddings and their corresponding preprocessed chunks to the vector store
add_to_vector_store(embeddings, preprocessed_chunks)

# Define a query text for which we want to retrieve relevant document chunks
query_text = "What is Quantum Computing?"

# Retrieve the most relevant chunks from the vector store based on the query text
relevant_chunks = retrieve_relevant_chunks(query_text)

# Print the first 50 characters of each retrieved relevant chunk
for idx, chunk in enumerate(relevant_chunks):
    print(f"Chunk {idx + 1}: {chunk[:50]} ... ")
    print("-" * 50)  # Print a separator line

  
