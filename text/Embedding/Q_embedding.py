## From https://huggingface.co/blog/embedding-quantization

# with q_embedding model
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
binary_embeddings = model.encode(
    ["I am driving to the lake.", "It is a beautiful day."],
    precision="binary",
)

# with quantize_embeddings
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
embeddings = model.encode(["I am driving to the lake.", "It is a beautiful day."])
binary_embeddings = quantize_embeddings(embeddings, precision="binary")

-----------------------------------------------------
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
corpus = load_dataset("nq_open", split="train[:1000]")["question"]
calibration_embeddings = model.encode(corpus)

embeddings = model.encode(["I am driving to the lake.", "It is a beautiful day."])
int8_embeddings = quantize_embeddings(
    embeddings,
    precision="int8",
    calibration_embeddings=calibration_embeddings,
)


## Ex. with BAAI/bge-m3 model
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Transformer, Pooling
from sentence_transformers.quantization import quantize_embeddings
import torch
import numpy as np
from typing import Sequence, Any
import time

class QuantizedSentenceTransformerEncoder:
    def __init__(self, model_name: str, device: str , quantization_bits: str | None = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name, device=self.device)
        self.quantization_bits = quantization_bits
       
        # Quantize the model
        #self.model.transformer = quantize_embeddings(self.model.transformer, self.quantization_bits)

    def encode(
        self, sentences: Sequence[str], *, prompt_name: str | None = None, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        # Generate embeddings
        print('??', sentences)
        embeddings = self.model.encode(sentences, convert_to_tensor=True, **kwargs)
       
        # If needed, you can apply additional quantization to the output embeddings here
        # For example:
        print(len(embeddings))
        st = time.time()
        quantized_embeddings= quantize_embeddings(embeddings, precision=self.quantization_bits)
        print(time.time()-st)
       
        return quantized_embeddings # Convert to numpy array for consistency

model = SentenceTransformer('BAAI/bge-m3', device=device)
text=['한 남자가 음식을 먹고 있다', '한 남자가 뭔가를 먹고 있다']

embeddings = model.encode(text)
quantized_embeddings= quantize_embeddings(embeddings, precision='int8')
