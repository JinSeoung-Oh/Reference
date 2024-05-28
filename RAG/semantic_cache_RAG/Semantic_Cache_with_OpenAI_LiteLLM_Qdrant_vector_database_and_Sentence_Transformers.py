## From https://pub.towardsai.net/semantic-caching-in-generative-ai-chatbots-b134f116a50b

"""
Create and activate a conda environment
conda create -n semantic_cache python=3.11
conda activate semantic_cache

# Install the necessary libraries
pip install -U fastapi uvicorn loguru pandas numpy tqdm
pip install -U litellm sentence-transformers 
pip install -U qdrant-client redisvl==0.0.7

# Run Qdrant Vector Database locally
docker run --rm -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage:z qdrant/qdrant
"""

import os
import time

import litellm
from fastapi import FastAPI, Response
from litellm.caching import Cache
from loguru import logger
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

litellm.openai_key = os.getenv("OPENAI_API_KEY")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Step 1: Embedding model
encoder = SentenceTransformer("sentence-transformers/stsb-mpnet-base-v2")

# Step 2: Vector DB to store the cache
collection_name = "semantic_cache"
qdrant_client = QdrantClient("localhost", port=6333)

try:
    qdrant_client.get_collection(collection_name=collection_name)
except:
    qdrant_client.recreate_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )

# Helper
def generate_embedding(**kwargs) -> list:
    """Take the last message as the prompt and generate an embedding
    Args:
        kwargs: All the arguments passed to the litellm call
    Returns:
        list: Embedding vector of the prompt
    """
    # Take the last message as the prompt
    prompt = kwargs.get("messages", [])[-1].get("content", "")

    # Embed
    litellm_embedding = encoder.encode(prompt).tolist()

    return litellm_embedding

# Step 3: Add Cache
def add_cache(result: litellm.ModelResponse, **kwargs) -> None:
    """Add the result to the cache
    Args:
        result (litellm.ModelResponse): Response from the litellm call
        kwargs: All the arguments passed to the litellm call 
        and some additional keys: 
        `litellm_call_id`, `litellm_logging_obj` and `preset_cache_key`
    """
    # Embed
    litellm_embedding = generate_embedding(**kwargs)

    # Upload to vector DB
    qdrant_client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=kwargs.get("litellm_call_id"),
                vector=litellm_embedding,
                payload=result.dict(),
            )
        ],
    )

# Step 4: Get Cache
def get_cache(**kwargs) -> dict:
    """Read the cache
    Args:
        kwargs: All the arguments passed to the litellm call 
        and some additional keys: 
        `litellm_call_id`, `litellm_logging_obj` and `preset_cache_key`
    Returns:
        dict: The result that was saved in the cache. 
        Should be compatible with litellm.ModelResponse schema
    """
    similarity_threshold = 0.95

    # Embed
    litellm_embedding = generate_embedding(**kwargs)

    # Cache Search
    hits = qdrant_client.search(
        collection_name=collection_name, 
        query_vector=litellm_embedding, 
        limit=5
    )

    # Similarity threshold
    similar_docs = [
        {**hit.payload, "score": hit.score}
        for hit in hits
        if hit.score > similarity_threshold
    ]

    if similar_docs:
        logger.info("Cache hit!")
    else:
        logger.info("Cache miss!")

    # Return result
    return similar_docs[0] if similar_docs else None

# Step 5: Semantic Cache
cache = Cache()
cache.add_cache = add_cache
cache.get_cache = get_cache
litellm.cache = cache

# Step 6: Fast API App
app = FastAPI()

@app.get("/")
def health_check():
    return {"Status": "Alive"}

@app.post("/chat")
def chat(question: str, response: Response) -> dict:
    # LiteLLM call - Handles the cache internally
    start = time.time()
    result = litellm.completion(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": question}],
        max_tokens=100,
    )
    end = time.time()

    # Add latency to the response header
    response.headers["X-Response-Time"] = str(end - start)

    return {"response": result.choices[0].message.content}

## Challenges & Benchmarking
import numpy as np
import pandas as pd
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder, SentenceTransformer
from tqdm import tqdm

# Embedding model
encoder = SentenceTransformer("stsb-mpnet-base-v2")

# Re-ranking model
ranker = CrossEncoder("cross-encoder/stsb-roberta-base")

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Corpus (500K questions)
corpus_df = pd.read_json("quora/corpus.jsonl", lines=True).reset_index(drop=True)
corpus_df = corpus_df.drop(columns=["title", "metadata"])
corpus_df = corpus_df.rename(columns={"_id": "corpus_id"})

# Queries (15K questions)
queries_df = pd.read_json("quora/queries.jsonl", lines=True).reset_index(drop=True)
queries_df = queries_df.drop(columns=["metadata"])
queries_df = queries_df.rename(columns={"_id": "query_id"})

# Indexing corpus into Qdrant Vector DB
collection_name = "quora"
if not collection_name in [item.name for item in client.get_collections().collections]:
    client.recreate_collection(
        collection_name="quora",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )

    # Upload documents
    batch_size = 16
    for batch in tqdm(range(0, corpus_df.shape[0], batch_size)):
        corpus_batch = corpus_df.iloc[batch : batch + batch_size].reset_index(drop=True)
        corpus_batch = corpus_batch.fillna("")
        vectors = encoder.encode(corpus_batch["text"]).tolist()

        client.upsert(
            collection_name=collection_name,
            points=models.Batch(
                ids=corpus_batch["corpus_id"].tolist(),
                vectors=vectors,
                payloads=corpus_batch.to_dict(orient="records"),
            ),
        )
else:
    print("Collection already exists")
    print(client.get_collection(collection_name=collection_name))

def cache_hits(queries_df, similarity_threshold=0.99):
    retriever_cache_hit = []
    retriever_ranker_cache_hit = []

    for row in tqdm(queries_df.itertuples(), total=len(queries_df)):
        # Semantic Search
        hits = client.search(
            collection_name=collection_name,
            query_vector=encoder.encode(row.text).tolist(),
            limit=20,
        )

        # Check for self match
        if row.query_id == hits[0].payload["corpus_id"]:
            hits = hits[1:]

        # Similarity threshold
        similar_docs = [
            {**hit.payload, "score": hit.score}
            for hit in hits
            if hit.score > similarity_threshold
        ]

        if len(similar_docs):
            best_match = similar_docs[0]
            retriever_cache_hit.append(best_match["corpus_id"])

            ranker_results = ranker.predict(
                [[row.text, doc["text"]] for doc in similar_docs]
            )

            best_match = similar_docs[np.argmax(ranker_results)]

            retriever_ranker_cache_hit.append(best_match["corpus_id"])

        else:
            retriever_cache_hit.append(None)
            retriever_ranker_cache_hit.append(None)

    queries_df[f"retriever_cache_hit_{similarity_threshold}"] = retriever_cache_hit
    queries_df[f"retriever_ranker_cache_hit_{similarity_threshold}"] = (
        retriever_ranker_cache_hit
    )

    return queries_df

def cache_hit_labels(row, col, test_df):
    actual_similar_docs = test_df[test_df["query_id"] == row["query_id"]][
        "corpus_id"
    ].values.tolist()

    if not pd.isnull(row[col]) and row[col]:
        if row[col] in actual_similar_docs:
            return "tp"
        else:
            return "fp"
    else:
        if len(actual_similar_docs):
            return "fn"
        else:
            return "tn"

def write_fp_df(labels, method, similarity_threshold):
    fp_df = queries_df.iloc[labels[labels == "fp"].index].copy()
    fp_df["corpus_text"] = fp_df[f"{method}_cache_hit_{similarity_threshold}"].apply(
        lambda x: corpus_df[corpus_df["corpus_id"] == x]["text"].values[0]
    )
    fp_df = fp_df.rename(
        columns={
            "text": "query_text",
            f"{method}_cache_hit_{similarity_threshold}": "corpus_id",
        }
    )
    fp_df["score"] = 0
    fp_df = fp_df[["query_id", "corpus_id", "score", "query_text", "corpus_text"]]
    fp_df["corpus_id"] = fp_df["corpus_id"].astype(int)
    fp_df.to_csv(
        f"../data/quora/fp/fp_{method}_{similarity_threshold}.csv", index=False
    )

def cache_hit_results(queries_df, test_df, similarity_threshold):
    print(f"Similarity threshold: {similarity_threshold}")

    cache_rates = {}
    for method in ["retriever", "retriever_ranker"]:
        print(f"Method: {method}")
        col = f"{method}_cache_hit_{similarity_threshold}"
        labels = queries_df.apply(
            lambda row: cache_hit_labels(row, col, test_df), axis=1
        )

        write_fp_df(labels, method, similarity_threshold)

        hit_counts = labels.value_counts()

        cache_rates[method] = {
            "tp": hit_counts.get("tp", 0),
            "fp": hit_counts.get("fp", 0),
            "fn": hit_counts.get("fn", 0),
            "tn": hit_counts.get("tn", 0),
        }

        cache_rates[method]["cache_hit_rate"] = round(
            cache_rates[method]["tp"]
            / (cache_rates[method]["tp"] + cache_rates[method]["fn"]),
            2,
        )
        cache_rates[method]["cache_precision"] = round(
            cache_rates[method]["tp"]
            / (cache_rates[method]["tp"] + cache_rates[method]["fp"]),
            2,
        )

        print(cache_rates[method])
    print("\n")

    return cache_rates

## Run 
import pandas as pd

import helpers

# Ground Truth of duplicate questions
test_df = pd.read_csv("quora/qrels/test.tsv", sep="\t").reset_index(drop=True)
dev_df = pd.read_csv("quora/qrels/dev.tsv", sep="\t").reset_index(drop=True)
test_df = pd.concat([test_df, dev_df], ignore_index=True)

test_df = test_df.rename(columns={"query-id": "query_id", "corpus-id": "corpus_id"})

# Load the "enhanced" hand labels file
hand_labels = pd.read_csv("quora/handlabels.csv").reset_index(drop=True)
test_df = pd.concat([test_df, dev_df], ignore_index=True)

# Run the benchmark
cache_hits_dict = {}

for similarity_threshold in [0.99, 0.95, 0.9]:
    queries_df = helpers.cache_hits(queries_df, similarity_threshold)
    cache_hits_dict[similarity_threshold] = helpers.cache_hit_results(queries_df, test_df, similarity_threshold)

print(cache_hits_dict)



