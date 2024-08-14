## From https://medium.com/towards-data-science/how-to-use-hybrid-search-for-better-llm-rag-retrieval-032f66810ebe

### Keyword Search With BM25

!pip install rank_bm25
from rank_bm25 import BM25Okapi

corpus = [
    "The cat, commonly referred to as the domestic cat or house cat, is a small domesticated carnivorous mammal.",
    "The dog is a domesticated descendant of the wolf.",
    "Humans are the most common and widespread species of primate, and the last surviving species of the genus Homo.",
    "The scientific name Felis catus was proposed by Carl Linnaeus in 1758"
]
tokenized_corpus = [doc.split(" ") for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
query = "The cat"
tokenized_query = query.split(" ")
doc_scores = bm25.get_scores(tokenized_query)
-------------------------------------------------------------------------------------------------------------------
### Semantic Search With Dense Embeddings

!pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
# The documents to encode
corpus = [
    "The cat, commonly referred to as the domestic cat or house cat, is a small domesticated carnivorous mammal.",
    "The dog is a domesticated descendant of the wolf.",
    "Humans are the most common and widespread species of primate, and the last surviving species of the genus Homo.",
    "The scientific name Felis catus was proposed by Carl Linnaeus in 1758"
]

# Calculate embeddings by calling model.encode()
document_embeddings = model.encode(corpus)
query = "The cat"
query_embedding = model.encode(query)
scores = cos_sim(document_embeddings, query_embedding)
-------------------------------------------------------------------------------------------------------------------
### Hybrid Search
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def scores_to_ranking(scores: list[float]) -> list[int]:
    """Convert float scores into int rankings (rank 1 is the best)"""
    return np.argsort(scores)[::-1] + 1

def rrf(keyword_rank: int, semantic_rank: int) -> float:
    """Combine keyword rank and semantic rank into a hybrid score."""
    k = 60
    rrf_score = 1 / (k + keyword_rank) + 1 / (k + semantic_rank)
    return rrf_score

def hybrid_search(
    query: str, corpus: list[str], encoder_model: SentenceTransformer
) -> list[int]:
    # bm25
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    tokenized_query = query.split(" ")
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_ranking = scores_to_ranking(bm25_scores)

    # embeddings
    document_embeddings = model.encode(corpus)
    query_embedding = model.encode(query)
    cos_sim_scores = cos_sim(document_embeddings, query_embedding).flatten().tolist()
    cos_sim_ranking = scores_to_ranking(cos_sim_scores)

    # combine rankings into RRF scores
    hybrid_scores = []
    for i, doc in enumerate(corpus):
        document_ranking = rrf(bm25_ranking[i], cos_sim_ranking[i])
        print(f"Document {i} has the rrf score {document_ranking}")
        hybrid_scores.append(document_ranking)

    # convert RRF scores into final rankings
    hybrid_ranking = scores_to_ranking(hybrid_scores)
    return hybrid_ranking
    
corpus = [
    "The cat, commonly referred to as the domestic cat or house cat, is a small domesticated carnivorous mammal.",
    "The dog is a domesticated descendant of the wolf.",
    "Humans are the most common and widespread species of primate, and the last surviving species of the genus Homo.",
    "The scientific name Felis catus was proposed by Carl Linnaeus in 1758",
]
query = "The cat"

bm25_ranking = [1, 2, 4, 3] # scores = [0.92932018 0.21121974 0. 0.1901173]
cosine_ranking = [1, 3, 4, 2] # scores = [0.5716, 0.2904, 0.0942, 0.3157]

hybrid_ranking = hybrid_search(
    query="What is the scientifc name for cats?", corpus=corpus, encoder_model=model
)
print(hybrid_ranking)
>> Document 0 has the rrf score 0.03125
>> Document 1 has the rrf score 0.032266458495966696
>> Document 2 has the rrf score 0.03225806451612903
>> Document 3 has the rrf score 0.032266458495966696
>> [4 2 3 1]



