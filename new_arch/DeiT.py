### From https://ai.gopubby.com/vision-transformer-on-a-budget-deit-3388fc2184cd

"""
1. Background & Problem
   -a. Vanilla ViT’s data hunger: The original Vision Transformer (ViT) achieves strong performance but only when trained on
       extremely large labeled datasets (e.g. JFT‑300M, ~300 million images). 
       This makes ViT prohibitively expensive in terms of data collection and computation.

2. Core Idea of DeiT: Knowledge Distillation
   -a. Teacher–Student Setup
       -1. Teacher: A pretrained RegNet CNN
       -2. Student: The DeiT (Data‑efficient image Transformer)
   -b. Training Phase:
       -1. The student not only learns from the ImageNet‑1K labels but also matches the teacher’s soft‑label predictions via 
           an added distillation loss term.
   -c. Inference Phase:
       -1. Only the DeiT student is used; the RegNet teacher is discarded.
   -d. Outcome:
       -1. Original ViT requires 300 M images → DeiT achieves competitive accuracy using just 1 M images (ImageNet‑1K), a 300× reduction in data.

3. Architectural Modification: Distillation Token
   -a. ViT: Uses a single class token prepended to the patch embeddings.
   -b. DeiT: Introduces an additional distillation token alongside the class token.
       -1. Both tokens are concatenated with patch embeddings, allowing the model to absorb both the classification and teacher‑matching signals.

4. DeiT vs. ViT Variants
   Variant	| Hidden Dim	| # Heads	| Depth	| Params (≈)
   DeiT‑Ti	| 192	 | 3	| 12	| 5 M
   DeiT‑S	| 384	| 6	| 12	| 22 M
   DeiT‑B	| 768	| 12	| 12	| 86 M
   ViT‑B	| 768	| 12	| 12	| 86 M

   -a. Key Point: DeiT‑B and ViT‑B share the same size, but DeiT‑B is data‑ and computation‑efficient.

5. DeiT‑B Detailed Configuration
   -a. Patch Embedding: Maps each image patch to a 768‑dimensional vector.
   -b. Multi‑Head Self‑Attention: 12 heads × 64‑dim per head = 768 dims.
   -c. Transformer Encoder: 12 layers of standard Transformer blocks.
   -d. Total Parameters: Approximately 86 million trainable weights.

6. Experimental Results on ImageNet‑1K
   -a. Benchmark Comparison: EfficientNet, ViT, DeiT, and two distilled variants (DeiT and DeiT⚗ “alembic”).
   -b. Findings:
       -1. Standard DeiT outperforms ViT when both are trained solely on ImageNet‑1K.
       -2. DeiT (with the novel distillation mechanism) plus fine‑tuning at 384×384 resolution (DeiT‑B↑384) achieves
           the highest accuracy without extra data.
   -c. Implication: ViT cannot match its full potential without massive pretraining data, whereas DeiT excels in data‑limited settings.
"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from torchinfo import summary

# Codeblock 2
BATCH_SIZE   = 1
IMAGE_SIZE   = 384     #(1)
IN_CHANNELS  = 3

PATCH_SIZE   = 16      #(2)
EMBED_DIM    = 768     #(3)
NUM_HEADS    = 12      #(4)
NUM_LAYERS   = 12      #(5)
FFN_SIZE     = EMBED_DIM * 4    #(6)

NUM_PATCHES  = (IMAGE_SIZE//PATCH_SIZE) ** 2    #(7)

NUM_CLASSES  = 1000    #(8)

class Patcher(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=IN_CHANNELS,    #(1)
                              out_channels=EMBED_DIM, 
                              kernel_size=PATCH_SIZE,     #(2)
                              stride=PATCH_SIZE)          #(3)

        self.flatten = nn.Flatten(start_dim=2)            #(4)

    def forward(self, x):
        print(f'original\t: {x.size()}')

        x = self.conv(x)        #(5)
        print(f'after conv\t: {x.size()}')

        x = self.flatten(x)     #(6)
        print(f'after flatten\t: {x.size()}')

        x = x.permute(0, 2, 1)  #(7)
        print(f'after permute\t: {x.size()}')

        return x
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.norm_0 = nn.LayerNorm(EMBED_DIM)    #(1)

        self.multihead_attention = nn.MultiheadAttention(EMBED_DIM,    #(2)
                                                         num_heads=NUM_HEADS, 
                                                         batch_first=True)

        self.norm_1 = nn.LayerNorm(EMBED_DIM)    #(3)

        self.ffn = nn.Sequential(                #(4)
            nn.Linear(in_features=EMBED_DIM, out_features=FFN_SIZE),
            nn.GELU(), 
            nn.Linear(in_features=FFN_SIZE, out_features=EMBED_DIM),
        )


    def forward(self, x):

        residual = x
        print(f'residual dim\t: {residual.size()}')

        x = self.norm_0(x)
        print(f'after norm\t: {x.size()}')

        x = self.multihead_attention(x, x, x)[0]
        print(f'after attention\t: {x.size()}')

        x = x + residual
        print(f'after addition\t: {x.size()}')

        residual = x
        print(f'residual dim\t: {residual.size()}')

        x = self.norm_1(x)
        print(f'after norm\t: {x.size()}')

        x = self.ffn(x)
        print(f'after ffn\t: {x.size()}')

        x = x + residual
        print(f'after addition\t: {x.size()}')

        return x
      

class DeiT(nn.Module):
    def __init__(self):
        super().__init__()

        self.patcher = Patcher()    #(1)
        
        self.class_token = nn.Parameter(torch.zeros(BATCH_SIZE, 1, EMBED_DIM))  #(2)
        self.dist_token  = nn.Parameter(torch.zeros(BATCH_SIZE, 1, EMBED_DIM))  #(3)
        
        trunc_normal_(self.class_token, std=.02)    #(4)
        trunc_normal_(self.dist_token, std=.02)     #(5)

        self.pos_embedding = nn.Parameter(torch.zeros(BATCH_SIZE, NUM_PATCHES+2, EMBED_DIM))  #(6)
        trunc_normal_(self.pos_embedding, std=.02)  #(7)
        
        self.encoders = nn.ModuleList([Encoder() for _ in range(NUM_LAYERS)])  #(8)
        
        self.norm_out = nn.LayerNorm(EMBED_DIM)     #(9)

        self.class_head = nn.Linear(in_features=EMBED_DIM, out_features=NUM_CLASSES)  #(10)
        self.dist_head  = nn.Linear(in_features=EMBED_DIM, out_features=NUM_CLASSES)  #(11)
      
    def forward(self, x):
        print(f'original\t\t: {x.size()}')
        
        x = self.patcher(x)           #(1)
        print(f'after patcher\t\t: {x.size()}')
        
        x = torch.cat([self.class_token, self.dist_token, x], dim=1)  #(2)
        print(f'after concat\t\t: {x.size()}')
        
        x = x + self.pos_embedding    #(3)
        print(f'after pos embed\t\t: {x.size()}')
        
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)            #(4)
            print(f"after encoder #{i}\t: {x.size()}")

        x = self.norm_out(x)          #(5)
        print(f'after norm\t\t: {x.size()}')
        
        class_out = x[:, 0]           #(6)
        print(f'class_out\t\t: {class_out.size()}')
        
        dist_out  = x[:, 1]           #(7)
        print(f'dist_out\t\t: {dist_out.size()}')
        
        class_out = self.class_head(class_out)    #(8)
        print(f'after class_head\t: {class_out.size()}')
        
        dist_out  = self.dist_head(dist_out)       #(9)
        print(f'after dist_head\t\t: {class_out.size()}')
        
        return class_out, dist_out
