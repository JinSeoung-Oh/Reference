### From https://ai.plainenglish.io/optimizing-rag-with-embedding-tuning-2508af2ec049

"""
1. Overview of RAG Embeddings
   In Retrieval-Augmented Generation (RAG) systems, embeddings are dense vector representations of texts. 
   Unlike sparse, high-dimensional one-hot encodings, embeddings capture semantic relationships in a continuous, low-dimensional space. 
   In a RAG system, embeddings are created for both user prompts and custom documents or domain-specific knowledge, 
   allowing for semantically aligned retrieval of relevant information based on the input prompt.

2. Embedding and Retrieval Workflow:

   -1. Embedding Generation
       The system embeds the user’s prompt and documents in the knowledge base. Models such as DPR, Sentence-BERT, 
       and RoBERTa are often used to create these embeddings.
   -2. Retrieval and Generation
       The embedded prompt is used to retrieve the closest matching data from the embedded knowledge base. 
       This information is then passed to the language model (e.g., GPT-4) to generate responses that align with the retrieved content.
   -3. Importance of Embedding Optimization
       Properly optimized embeddings ensure that the retrieved data is contextually relevant and enhances overall system performance.
       Pre-trained models may not fully capture specific domain nuances, 
       so fine-tuning embeddings to fit the domain improves accuracy and relevance in the RAG system.

3. Embedding Tuning Techniques:

   -1. Domain Adaptation: Tuning embeddings with domain-specific data (e.g., legal or medical terminology) improves relevance in specialized fields.
   -2. Contrastive Learning: Groups related queries and answers closer together while distancing unrelated ones, enhancing retrieval precision.
   -3. Supervised Signal Integration: Incorporates real-world feedback or labeled data to guide embeddings toward more useful retrieval patterns.
   -4. Self-Supervised Learning: Useful for systems with limited labeled data, allowing the model to find patterns within data for general-use RAG.
   -5. Combining Embeddings: Blends general-purpose with domain-specific embeddings to cover a broader range of topics.
   -6. Regularization: Techniques like dropout or triplet loss prevent overfitting on specific words or concepts, maintaining a balanced understanding.
   -7. Hard Negatives: Training with challenging examples refines the model's ability to differentiate correct from near-correct answers.
   -8. Feedback Loops: Active learning flags difficult prompts for human review, improving accuracy through iterative refinement.
   -9. Cross-Encoder Tuning: Matches queries and documents directly for higher semantic alignment in cases requiring close matching.

4. Evaluating Embedding Quality:

   -1. Cosine Similarity and Nearest Neighbor Evaluation: Measures similarity between embeddings for accurate retrieval.
   -2. Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP): Ranks retrieved documents by relevance.
   -3. Embedding Visualization: Uses tools like t-SNE or UMAP to visualize embedding clusters.
   -4. Human Judgment and Feedback: Human evaluations refine retrieval relevance.
   -5. Domain-Specific Metrics: Metrics tailored to domain-specific nuances ensure effectiveness in specialized fields.

5. Challenges in Embedding Tuning:

   -1. Computational Cost: Tuning embeddings is resource-intensive.
   -2. Overfitting: Risk of the model focusing too narrowly on training data, limiting retrieval accuracy on new prompts.
   -3. Data Quality: High-quality domain-specific data is essential for reliable embedding optimization.
   -4. Domain Shifts: Domains evolve, requiring regular updates to the model.

6. Conclusion
   RAG optimization is essential for developing high-accuracy systems. 
   Embedding tuning enhances retrieval precision and aligns model responses with user expectations. 
   Fine-tuning methods enable iterative improvements, ensuring the RAG system maintains relevance and accuracy across domains.
"""

import faiss  # This handles similarity search
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
 
# A pre-trained embedding model (e.g., BERT) is loaded
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
 
# Function to encode text into embeddings
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.cpu().numpy()
 
# Sample document corpus also known as the
# authoritative knowledge base, in this example it is for a # bakery shop
documents = [
    "We are open for 6 days of the week, on Monday, Tuesday, Wednesday, Thursday, Friday, Saturday",
    "The RAG system uses a retrieval component to fetch information.",
"We are located in Lagos, our address is 456 computer Lekki-Epe Express way.",
"Our CEO is Mr Austin, his phone number is 09090909090"]
 
# Encode documents and store in FAISS index
dimension = 384  # Set embedding dimension based on #the model used
index = faiss.IndexFlatL2(dimension)  # Create FAISS index
 
# Create document embeddings and add to FAISS index
doc_embeddings = np.vstack([embed_text(doc) for doc in documents])
index.add(doc_embeddings)
 
# Query Given it by a user
query = "Where is the location of your business?"
query_embedding = embed_text(query)
 
# Retrieve top 2 documents based on similarity
top_k = 2
_, indices = index.search(query_embedding, top_k)
retrieved_docs = [documents[idx] for idx in indices[0]]
 
print("Your Query:", query)
print("Retrieved Documents:", retrieved_docs)


#### Generation Component
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
import faiss
import torch

# Load Sentence Transformer model for embeddings (using PyTorch)
embed_model = SentenceTransformer('all-MiniLM-L6-v2')
# Sample documents for retrieval
documents = [
    "We are open for 6 days of the week, on Monday, Tuesday, Wednesday,            Thursday, Friday, Saturday",
    "The RAG system uses a retrieval component to fetch information.",
"We are located in Lagos, our address is 456 computer Lekki-Epe Express way.",
"Our CEO is Mr. Austin, his phone number is 09090909090"
]
# Embed the documents
doc_embeddings = embed_model.encode(documents)
# Use FAISS for fast similarity search
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)
# Load T5 model and tokenizer for the generation component
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")
# Define a query
query = "How does a RAG system work in machine learning?"
# Retrieve top-k relevant documents
query_embedding = embed_model.encode([query])
top_k = 2
_, indices = index.search(query_embedding, top_k)
retrieved_docs = [documents[idx] for idx in indices[0]]
# Concatenate retrieved docs to augment the query
augmented_query = query + " " + " ".join(retrieved_docs)
print("Augmented Query:", augmented_query)
# Prepare input for T5 model
input_text = f"answer_question: {augmented_query}"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
# Generate answer using T5
with torch.no_grad():
    output = model.generate(input_ids, max_length=50, num_beams=5, early_stopping=True)
answer = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated Answer:", answer)




