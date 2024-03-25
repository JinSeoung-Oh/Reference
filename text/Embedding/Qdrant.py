# From https://medium.com/@shaikhrayyan123/qdrant-using-fastembed-for-rapid-embedding-generation-a-benchmark-and-guide-dc105252c399

! pip install qdrant-client
! pip install -U sentence-transformers
! pip install fastembed

from qdrant_client import QdrantClient
from qdrant_client.http import models

# Connect to Qdrant
client = QdrantClient(host="localhost", port=6333)

# Benchmark 1: Using Qdrant without FastEmbed
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import time
# Load a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
# Sample dataset
sentences = ["This is a sample sentence.", "Embeddings are useful.", "Sentence for the embedding"] # Add more sentences
# Generate embeddings
start_time = time.time()
embeddings = model.encode(sentences)
end_time = time.time()
print("Time taken to generate embeddings:", end_time - start_time, "seconds")
vector_param = VectorParams(size=len(embeddings[0]), distance=Distance.DOT)
client.create_collection(collection_name=collection_name, vectors_config= vector_param)
client.upload_collection(collection_name=collection_name, vectors=embeddings)
# Perform a search query
query_vector = embeddings[0] # using the first sentence embedding as a query
search_results = client.search(collection_name=collection_name, query_vector=query_vector, top=5)
print(search_results)

# Benchmark 2: Using Qdrant with FastEmbed
from fastembed.embedding import DefaultEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import time
# Load a FastEmbed model
fastembed_model = DefaultEmbedding()
# Same dataset as the first benchmark
sentences = ["This is a sample sentence.", "Embeddings are useful."] # more sentences
# Generate embeddings with FastEmbed
start_time = time.time()
fast_embeddings = fastembed_model.embed(sentences)
end_time = time.time()
print("Time taken to generate embeddings with FastEmbed:", end_time - start_time, "seconds")
# Connect to Qdrant and upload FastEmbed embeddings
client = QdrantClient(host='localhost', port=6333)
collection_name = 'fastembed_collection'
vector_param = VectorParams(size=len(embeddings[0]), distance=Distance.DOT)
client.create_collection(collection_name=collection_name, vectors_config= vector_param)
client.upload_collection(collection_name=collection_name, vectors=fast_embeddings)
# Perform a search query
query_vector = fast_embeddings[0] # using the first sentence embedding as a query
search_results = client.search(collection_name=collection_name, query_vector=query_vector, top=5)
print(search_results)
