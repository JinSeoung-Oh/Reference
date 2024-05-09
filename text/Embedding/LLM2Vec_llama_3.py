# From https://towardsdatascience.com/turn-llama-3-into-an-embedding-model-with-llm2vec-8448005f99aa
# From https://huggingface.co/kaitchup/Llama-3-8B-llm2vec-Emb

############ Training Embedding model #################
pip install llm2vec
pip install flash-attn --no-build-isolation

"""
check : https://github.com/McGill-NLP/llm2vec/blob/main/experiments/run_mntp.py
and check : https://towardsdatascience.com/turn-llama-3-into-an-embedding-model-with-llm2vec-8448005f99aa
"""

import torch
from llm2vec import LLM2Vec

l2v = LLM2Vec.from_pretrained(
    "meta-llama/Meta-Llama-3-8B",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

l2v.save("Llama-3-8B-Emb")

#########################################################


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("kaitchup/Llama-3-8B-llm2vec-Emb")
# (or) Settings.embed_model = HuggingFaceEmbedding(model_name="kaitchup/Llama-3-8B-llm2vec-Emb", device='cpu') -- If use LlamaIndex
