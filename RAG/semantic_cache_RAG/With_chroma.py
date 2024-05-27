## From https://huggingface.co/learn/cookbook/semantic_cache_chroma_vector_database

# 1. semantic_cache_chroma_vector_database
!pip install -q transformers==4.38.1
!pip install -q accelerate==0.27.2
!pip install -q sentence-transformers==2.5.1
!pip install -q xformers==0.0.24
!pip install -q chromadb==0.4.24
!pip install -q datasets==2.17.1

import numpy as np
import pandas as pd
from getpass import getpass
import chromadb

if 'hf_key' not in locals():
  hf_key = getpass("Your Hugging Face API Key: ")
!huggingface-cli login --token $hf_key

from datasets import load_dataset
data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
data = data.to_pandas()
data["id"] = data.index

MAX_ROWS = 15000
DOCUMENT = "Answer"
TOPIC = "qtype"

###
def query_database(query_text, n_results=10):
    results = collection.query(query_texts=query_text, n_results=n_results)
    return results

subset_data = data.head(MAX_ROWS)
chroma_client = chromadb.PersistentClient(path="/path/to/persist/directory")
collection_name = "news_collection"
if len(chroma_client.list_collections()) > 0 and collection_name in [chroma_client.list_collections()[0].name]:
    chroma_client.delete_collection(name=collection_name)

collection = chroma_client.create_collection(name=collection_name)

collection.add(
    documents=subset_data[DOCUMENT].tolist(),
    metadatas=[{TOPIC: topic} for topic in subset_data[TOPIC].tolist()],
    ids=[f"id{x}" for x in range(MAX_ROWS)],
)


#### !pip install -q faiss-cpu==1.8.0
import faiss
from sentence_transformers import SentenceTransformer
import time
import json

def init_cache():
    index = faiss.IndexFlatL2(768)
    if index.is_trained:
        print("Index trained")

    # Initialize Sentence Transformer model
    encoder = SentenceTransformer("all-mpnet-base-v2")

    return index, encoder

def retrieve_cache(json_file):
    try:
        with open(json_file, "r") as file:
            cache = json.load(file)
    except FileNotFoundError:
        cache = {"questions": [], "embeddings": [], "answers": [], "response_text": []}

    return cache

def store_cache(json_file, cache):
    with open(json_file, "w") as file:
        json.dump(cache, file)

class semantic_cache:
    def __init__(self, json_file="cache_file.json", thresold=0.35):
        # Initialize Faiss index with Euclidean distance
        self.index, self.encoder = init_cache()

        # Set Euclidean distance threshold
        # a distance of 0 means identicals sentences
        # We only return from cache sentences under this thresold
        self.euclidean_threshold = thresold

        self.json_file = json_file
        self.cache = retrieve_cache(self.json_file)

    def ask(self, question: str) -> str:
        # Method to retrieve an answer from the cache or generate a new one
        start_time = time.time()
        try:
            # First we obtain the embeddings corresponding to the user question
            embedding = self.encoder.encode([question])

            # Search for the nearest neighbor in the index
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)

            if D[0] >= 0:
                if I[0][0] >= 0 and D[0][0] <= self.euclidean_threshold:
                    row_id = int(I[0][0])

                    print("Answer recovered from Cache. ")
                    print(f"{D[0][0]:.3f} smaller than {self.euclidean_threshold}")
                    print(f"Found cache in row: {row_id} with score {D[0][0]:.3f}")
                    print(f"response_text: " + self.cache["response_text"][row_id])

                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    return self.cache["response_text"][row_id]

            # Handle the case when there are not enough results
            # or Euclidean distance is not met, asking to chromaDB.
            answer = query_database([question], 1)
            response_text = answer["documents"][0][0]

            self.cache["questions"].append(question)
            self.cache["embeddings"].append(embedding[0].tolist())
            self.cache["answers"].append(answer)
            self.cache["response_text"].append(response_text)

            print("Answer recovered from ChromaDB. ")
            print(f"response_text: {response_text}")

            self.index.add(embedding)
            store_cache(self.json_file, self.cache)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")

            return response_text
        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")


## !pip install torch
from torch import cuda, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = f"cuda:{cuda.current_device()}" if cuda.is_available() else "cpu"
model_id = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", torch_dtype=torch.bfloat16)


### Creating the extended prompt
prompt_template = f"Relevant context: {results}\n\n The user's question: {question_def}"
input_ids = tokenizer(prompt_template, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=256)




