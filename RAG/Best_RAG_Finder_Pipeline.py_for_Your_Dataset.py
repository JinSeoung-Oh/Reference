### From https://levelup.gitconnected.com/creating-the-best-rag-finder-pipeline-for-your-dataset-88062a6fa45e

# Install libraries (run this cell only once if needed)
!pip install openai pandas numpy faiss-cpu ipywidgets tqdm scikit-learn

import os                     # For accessing environment variables (like API keys)
import time                   # For timing operations
import re                     # For regular expressions (text cleaning)
import warnings               # For controlling warning messages
import itertools              # For creating parameter combinations easily
import getpass                # For securely prompting for API keys if not set

import numpy as np            # Numerical library for vector operations
import pandas as pd           # Data manipulation library for tables (DataFrames)
import faiss                  # Library for fast vector similarity search
from openai import OpenAI     # Client library for Nebius API interaction
from tqdm.notebook import tqdm # Library for displaying progress bars
from sklearn.metrics.pairwise import cosine_similarity # For calculating similarity score

# If using a local model like Ollama
OPENAI_API_KEY='ollama' # Can be any non-empty string for Ollama
OPENAI_API_BASE='http://localhost:11434/v1'

# --- NebiusAI API Configuration ---
# BEST PRACTICE: Use environment variables or a secure method for API keys!
NEBIUS_API_KEY = os.getenv('NEBIUS_API_KEY') # <-- *** SET YOUR KEY SAFELY ***

NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1/"
NEBIUS_EMBEDDING_MODEL = "BAAI/bge-multilingual-gemma2"  # For text-to-vector conversion
NEBIUS_GENERATION_MODEL = "deepseek-ai/DeepSeek-V3"    # For generating final answers
NEBIUS_EVALUATION_MODEL = "deepseek-ai/DeepSeek-V3"    # For evaluating the generated answers

# --- Text Generation Parameters (for the final answer) ---
GENERATION_TEMPERATURE = 0.1  # Low temp for factual, focused answers
GENERATION_MAX_TOKENS = 400   # Max answer length
GENERATION_TOP_P = 0.9        # Usually fine at default

# Create the OpenAI client object, configured for the Nebius API.
client = OpenAI(
    api_key=NEBIUS_API_KEY,     # Pass the API key loaded earlier
    base_url=NEBIUS_BASE_URL  # Specify the Nebius API endpoint
)

# --- Parameters to Tune ---
CHUNK_SIZES_TO_TEST = [150, 250]    # List of chunk sizes (in words) to experiment with.
CHUNK_OVERLAPS_TO_TEST = [30, 50]   # List of chunk overlaps (in words) to experiment with.
RETRIEVAL_TOP_K_TO_TEST = [3, 5]   # List of 'k' values (number of chunks to retrieve) to test.

# --- Reranking Configuration ---
# For simulated reranking: retrieve K * multiplier chunks initially.
# A real reranker would then re-score these based on relevance.
RERANK_RETRIEVAL_MULTIPLIER = 3

# --- Evaluation Prompts (Instructions for the NEBIUS_EVALUATION_MODEL LLM) ---
# 1. Faithfulness: Does the answer accurately reflect the True Answer / context?
FAITHFULNESS_PROMPT = """
System: You are an objective evaluator. Evaluate the faithfulness of the AI Response compared to the True Answer, considering only the information present in the True Answer as the ground truth.
Faithfulness measures how accurately the AI response reflects the information in the True Answer, without adding unsupported facts or contradicting it.
Score STRICTLY using a float between 0.0 and 1.0, based on this scale:
- 0.0: Completely unfaithful, contradicts or fabricates information.
- 0.1-0.4: Low faithfulness with significant inaccuracies or unsupported claims.
- 0.5-0.6: Partially faithful but with noticeable inaccuracies or omissions.
- 0.7-0.8: Mostly faithful with only minor inaccuracies or phrasing differences.
- 0.9: Very faithful, slight wording differences but semantically aligned.
- 1.0: Completely faithful, accurately reflects the True Answer.
Respond ONLY with the numerical score.

User:
Query: {question}
AI Response: {response}
True Answer: {true_answer}
Score:"""

# 2. Relevancy: Does the answer directly address the user's specific query?
RELEVANCY_PROMPT = """
System: You are an objective evaluator. Evaluate the relevance of the AI Response to the specific User Query.
Relevancy measures how well the response directly answers the user's question, avoiding unnecessary or off-topic information.
Score STRICTLY using a float between 0.0 and 1.0, based on this scale:
- 0.0: Not relevant at all.
- 0.1-0.4: Low relevance, addresses a different topic or misses the core question.
- 0.5-0.6: Partially relevant, answers only a part of the query or is tangentially related.
- 0.7-0.8: Mostly relevant, addresses the main aspects of the query but might include minor irrelevant details.
- 0.9: Highly relevant, directly answers the query with minimal extra information.
- 1.0: Completely relevant, directly and fully answers the exact question asked.
Respond ONLY with the numerical score.

User:
Query: {question}
AI Response: {response}
Score:"""

def chunk_text(text, chunk_size, chunk_overlap):
    """Splits a single text document into overlapping chunks based on word count.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The target number of words per chunk.
        chunk_overlap (int): The number of words to overlap between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """

    words = text.split()      # Split the text into a list of individual words
    total_words = len(words) # Calculate the total number of words in the text
    chunks = []             # Initialize an empty list to store the generated chunks
    start_index = 0         # Initialize the starting word index for the first chunk

    # --- Input Validation ---
    if chunk_overlap >= chunk_size:
        adjusted_overlap = chunk_size // 3
        print(f"  Warning: overlap ({chunk_overlap}) >= size ({chunk_size}). Adjusting to {adjusted_overlap}.")
        chunk_overlap = adjusted_overlap

    # --- Chunking Loop ---
    while start_index < total_words:
        # Determine end index, careful not to go beyond the total word count
        end_index = min(start_index + chunk_size, total_words)
        # Join the words for the current chunk
        current_chunk_text = " ".join(words[start_index:end_index])
        chunks.append(current_chunk_text)

        # Calculate the start of the next chunk
        next_start_index = start_index + chunk_size - chunk_overlap

        # Move to the next starting position
        start_index = next_start_index

    return chunks # Return the list of chunks

def calculate_cosine_similarity(text1, text2, client, embedding_model):
    """Calculates cosine similarity between the embeddings of two texts.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.
        client (OpenAI): The initialized Nebius AI client.
        embedding_model (str): The name of the embedding model to use.

    Returns:
        float: The cosine similarity score (between 0.0 and 1.0)
    """

    # Generate embeddings for both texts in a single API call
    response = client.embeddings.create(model=embedding_model, input=[text1, text2])

    # Extract the embedding vectors as NumPy arrays
    embedding1 = np.array(response.data[0].embedding)
    embedding2 = np.array(response.data[1].embedding)

    # Reshape vectors to be 2D arrays as expected by scikit-learn's cosine_similarity
    embedding1 = embedding1.reshape(1, -1)
    embedding2 = embedding2.reshape(1, -1)

    # Calculate cosine similarity using scikit-learn
    # Result is a 2D array like [[similarity]], so extract the value [0][0]
    similarity_score = cosine_similarity(embedding1, embedding2)[0][0]

    # Clamp the score between 0.0 and 1.0 for consistency
    return similarity_score

def run_and_evaluate(strategy_name, query_text, top_k, use_rerank=False):
    # Initialize result
    result = {
        'chunk_size': last_chunk_size, 'overlap': last_overlap, 'top_k': top_k,
        'strategy': strategy_name, 'retrieved_indices': [], 'rewritten_query': None,
        'answer': '', 'faithfulness': 0.0, 'relevancy': 0.0,
        'similarity_score': 0.0, 'avg_score': 0.0, 'time_sec': 0.0
    }

    start_time = time.time()

    # Embed query
    query_emb = client.embeddings.create(model=NEBIUS_EMBEDDING_MODEL, input=[query_text]).data[0].embedding
    query_vec = np.array([query_emb]).astype('float32')

    # Retrieve documents
    k_search = top_k * RERANK_RETRIEVAL_MULTIPLIER if use_rerank else top_k
    distances, indices = current_index.search(query_vec, min(k_search, current_index.ntotal))
    retrieved = indices[0][indices[0] != -1][:top_k]
    result['retrieved_indices'] = list(retrieved)

    # Prepare prompt
    retrieved_chunks = [current_chunks[i] for i in retrieved]
    context = "\n\n".join(retrieved_chunks)
    user_prompt = f"Context:\n------\n{context}\n------\n\nQuery: {test_query}\n\nAnswer:"
    sys_prompt = "You are a helpful assistant. Answer based only on the provided context."

    # Generate answer
    response = client.chat.completions.create(
        model=NEBIUS_GENERATION_MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=GENERATION_TEMPERATURE,
        max_tokens=GENERATION_MAX_TOKENS,
        top_p=GENERATION_TOP_P
    )
    result['answer'] = response.choices[0].message.content.strip()

    # Evaluate answer
    eval_params = {'model': NEBIUS_EVALUATION_MODEL, 'temperature': 0.0, 'max_tokens': 10}

    faith_prompt = FAITHFULNESS_PROMPT.format(question=test_query, response=result['answer'], true_answer=true_answer_for_query)
    result['faithfulness'] = float(client.chat.completions.create(messages=[{"role": "user", "content": faith_prompt}], **eval_params).choices[0].message.content.strip())

    relevancy_prompt = RELEVANCY_PROMPT.format(question=test_query, response=result['answer'])
    result['relevancy'] = float(client.chat.completions.create(messages=[{"role": "user", "content": relevancy_prompt}], **eval_params).choices[0].message.content.strip())

    result['similarity_score'] = calculate_cosine_similarity(result['answer'], true_answer_for_query, client, NEBIUS_EMBEDDING_MODEL)
    result['avg_score'] = (result['faithfulness'] + result['relevancy'] + result['similarity_score']) / 3.0

    result['time_sec'] = time.time() - start_time

    print(f"{strategy_name} (C={last_chunk_size}, O={last_overlap}, K={top_k}) done. Avg Score: {result['avg_score']:.2f}")

    return result

# Results container
all_results = []

# Test all combinations
param_combinations = list(itertools.product(CHUNK_SIZES_TO_TEST, CHUNK_OVERLAPS_TO_TEST, RETRIEVAL_TOP_K_TO_TEST))

for chunk_size, chunk_overlap, top_k in tqdm(param_combinations, desc="Running experiments"):

    # (Re)build the index if needed
    prepare_index(chunk_size, chunk_overlap)
    if not current_index:
        continue

    # Strategy 1: Simple RAG
    all_results.append(run_and_evaluate("Simple RAG", test_query, top_k))

    # Strategy 2: Query Rewrite RAG
    sys_prompt_rw = "Rewrite the user's query for better retrieval. Focus on keywords."
    user_prompt_rw = f"Original Query: {test_query}\n\nRewritten Query:"
    response = client.chat.completions.create(
        model=NEBIUS_GENERATION_MODEL,
        messages=[{"role": "system", "content": sys_prompt_rw},
                  {"role": "user", "content": user_prompt_rw}],
        temperature=0.1, max_tokens=100, top_p=0.9
    )
    rewritten_query = re.sub(r'^(rewritten query:|query:)\s*', '', response.choices[0].message.content.strip(), flags=re.IGNORECASE)
    if rewritten_query.lower() != test_query.lower():
        all_results.append(run_and_evaluate("Query Rewrite RAG", rewritten_query, top_k))
    
    # Strategy 3: Rerank RAG
    all_results.append(run_and_evaluate("Rerank RAG (Simulated)", test_query, top_k, use_rerank=True))

print("All experiments completed!")




