### From https://github.com/styalai/LCM-torch/blob/main/LCM/LCM.py
### From https://pub.towardsai.net/implement-a-large-concept-model-with-pytorch-5244c16a8cc0

import torch
import torch.nn as nn
from wtpsplit import SaT
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline

class Transformer(nn.Module):
    def __init__(self, embd_dim, dim, layers, heads, device):
        super().__init__()
        self.embd_dim = embd_dim
        self.dim = dim
        self.layers = layers
        self.heads = heads
        
        self.prenet = nn.Sequential(
            nn.LayerNorm(embd_dim),
            nn.Linear(embd_dim, dim)
        )

        self.decoder = nn.ModuleList([nn.TransformerDecoderLayer(d_model=dim, nhead=heads) for i in range(layers)])
        
        self.postnet = nn.Sequential(
            nn.Linear(dim, embd_dim),
            nn.LayerNorm(embd_dim)
        )
    def forward(self, x):
        x = self.prenet(x)
        for l in self.decoder:
            x = l(x, x)
        return self.postnet(x)


class LCMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sat_sm = SaT(config.model_name)
        print("splitter initialized")
        
        self.t2vec_model = TextToEmbeddingModelPipeline(encoder=config.sonar_enc, tokenizer=config.sonar_enc, device=torch.device(config.device))
        print("t2vec_model initialized")
        
        self.transformer = Transformer(config.embd_dim, config.dim, config.layers, config.heads, config.device).to(config.device)
        print("transformer initialized")
        
        self.vec2text_model = EmbeddingToTextModelPipeline(decoder=config.sonar_dec, tokenizer=config.sonar_dec, device=torch.device(config.device))
        print("vec2text_model initialized")
    
    def split_into_concepts(self, text):
        return self.sat_sm.split(text, threshold=self.config.threshold)
    
    def forward(self, embeddings):
        out_embeddings = self.transformer.forward(embeddings)
        return out_embeddings
    
    def generate(self, text, num_generated_concepts=1):
        with torch.no_grad():
            concepts = self.split_into_concepts(text)
            for c in range(num_generated_concepts):
                embeddings = self.t2vec_model.predict(concepts, source_lang=self.config.lang)
                out_embeddings = self.forward(embeddings)
                next_concept = self.vec2text_model.predict(out_embeddings, target_lang=self.config.lang, max_seq_len=self.config.max_seq_len)
                concepts.append(next_concept[0])
        return "".join(concepts)
