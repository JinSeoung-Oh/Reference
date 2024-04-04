"""
Retrieval Augmented Fine Tuning (RAFT) provide a means to infuse LLMs with domain-specific knowledge and reasoning abilities.

The integration of RAFT with LlamaIndex offers numerous advantages:

1. Enhanced Adaptability: Fine-tuning LLMs with domain-specific documents through RAFT enhances their understanding of specialized topics, 
                          thereby increasing adaptability in nuanced environments.

2. Improved Reasoning: RAFT enables LLMs to discern relevant information from retrieved documents, 
                       leading to more accurate and contextually appropriate responses.

3. Robustness Against Inaccurate Retrievals: RAFT trains LLMs to comprehend the relationship between the question, 
                                             retrieved documents, and the answer, ensuring resilience against inaccuracies in the retrieval process.

4. Efficient Knowledge Integration: By simulating real-world scenarios where LLMs must utilize external sources for information, 
                                    RAFT streamlines the integration of domain-specific knowledge into the model's framework, 
                                    resulting in more efficient knowledge utilization.
"""

# Step I: Install Libraries and Download Data
!pip install llama-index
!pip install llama-index-packs-raft-dataset

# Step II: Download RAFT Pack

import os

## Have to check RAFTDatasetPack
from llama_index.packs.raft_dataset import RAFTDatasetPack

os.environ["OPENAI_API_KEY"] = "<YOUR OPENAI API KEY>"

raft_dataset = RAFTDatasetPack("./paul_graham_essay.txt")

dataset = raft_dataset.run()
