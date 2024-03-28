"""
From https://instructor-embedding.github.io/
INSTRUCTOR, a novel approach for computing text embeddings based on task instructions, 
representing a significant advancement in natural language processing (NLP). INSTRUCTOR generates task-specific embeddings efficiently 
and without the need for additional training, setting new benchmarks in versatility and performance.

Key aspects of INSTRUCTOR include its integration of task instructions into the embedding process, 
its utilization of Generalized Text Representation (GTR) models as a backbone, and its training on the MEDI dataset comprising diverse tasks and instructions.

The training objective of INSTRUCTOR involves a text-to-text problem formulation, 
teaching the model to distinguish between good and bad outputs within the context of task-specific instructions.
Standardization of instructions across tasks ensures consistency and enhances the model's adaptability.

INSTRUCTOR outperforms traditional models across a wide range of tasks, 
showcasing an average performance enhancement of 3.4% over 70 diverse datasets. 
Despite its smaller size, it exhibits robustness and efficiency, heralding a new era in NLP.

The text also provides code snippets for integrating and utilizing INSTRUCTOR for various NLP tasks, 
demonstrating its ease of use and versatility. Overall, INSTRUCTOR represents a groundbreaking advancement in text embedding technology, 
promising exciting developments for NLP applications and research.
"""

## Seamless Embedding Generation
from transformers import AutoTokenizer, AutoModel
import torch
# Load tokenizer and model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("hkunlp/instructor-large")
model = AutoModel.from_pretrained("hkunlp/instructor-large")
# Your text and instructions go here
text = "One Embedder to rule them all, One Embedder to find them."
instructions = "Generate an embedding for a general understanding."
# Prepare input
inputs = tokenizer(text, instructions, return_tensors="pt")
# Generate embeddings
with torch.no_grad():
    embeddings = model(**inputs).last_hidden_state.mean(dim=1)
print(embeddings)

