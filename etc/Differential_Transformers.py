### From https://pub.towardsai.net/a-new-approach-to-attention-differential-transformers-paper-walkthrough-and-pytorch-15389743ff5b

"""
1. Core Concept
   -a. Motivation:
       Traditional transformers compute attention scores for every token in a sequence, 
       which can lead to irrelevant tokens receiving undue weight (known as attention noise). 
       The Differential Transformer addresses this by borrowing an idea from active noise cancellation (ANC) 
       used in headphones: cancel out the common (noisy) parts of the signal so that the desired information stands out.
    -b. Key Idea:
        Instead of generating a single pair of query (Q) and key (K) vectors, the model splits these into two halves. 
        It then computes two separate attention maps:
        -1. A₁: Calculated using the first half (Q₁ and K₁)
        -2. A₂: Calculated using the second half (Q₂ and K₂)
        By subtracting a scaled version of the second attention map from the first—formally, computing
        DiffAttn = (A₁ − λ·A₂) V—the model effectively cancels out the common noise, 
        sharpening its focus on the most relevant parts of the input.

2. How It Works
   -a. Splitting Q and K Vectors:
       -1. The input X is projected into query and key vectors.
       -2. These vectors are then split into two equal parts:
           Q → Q₁ and Q₂
           K → K₁ and K₂
   -b. Separate Attention Computation:
       -1. Each pair (Q₁ with K₁ and Q₂ with K₂) undergoes the standard attention operation:
           -1) Compute dot products.
           -2) Scale by the square root of half the head dimension.
           -3) Apply softmax to get normalized attention maps, A₁ and A₂.
   -c. Differential Attention Mechanism:
       -1. The final attention output is obtained by subtracting the second attention map from the first, 
           scaled by a learnable parameter λ:
           -1) DiffAttn = (A₁ − λ·A₂) V
       -2. This subtraction is designed to cancel out the shared noise (irrelevant tokens), 
           enhancing the prominence of useful signals.
   -d. Learnable Damping Factor (λ):
       -1. λ acts as a damping factor, balancing the contribution of the two attention maps.
       -2. It’s initialized based on the model’s depth (for example, using a formula like 
           λ_init = 0.8 − 0.6 × exp(−0.3 · depth)) and then learned during training.
       -3. This adaptive factor ensures that the noise cancellation does not overly suppress tokens that are 
           important.
   -e. Stability and Quantization Benefits:
       -1. By subtracting similar (noisy) components, the differential mechanism stabilizes the activations 
           in the attention scores, preventing extreme values that can distort the softmax output.
       -2. Experimentally, this approach has shown that a Differential Transformer with 4-bit quantization can 
           outperform a standard transformer even when the latter uses 6-bit quantization, 
           highlighting its efficiency.
"""

import torch.nn as nn
import torch.nn.functional as f
from apex.normalization import FusedRMSNorm as RMSNorm
import math 


class DiffAttn(nn.Module):
    def __init__(self, num_heads, embed_dim, depth):
        super().__init__()
        self.head_dim = int(embed_dim/num_heads)
        
        self.q_linear = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.k_linear = nn.Linear(embed_dim, self.head_dim, bias=False)
        self.v_linear = nn.Linear(embed_dim, self.head_dim, bias=False)

        self.lambda_init = 0.8 - 0.6 * math.exp(-0.3 * depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim//2, dtype=torch.float32))
        
        # mean = 0 (default); std = 0.1
        nn.init.normal_(self.lambda_q1, std=0.1)
        nn.init.normal_(self.lambda_q2, std=0.1)
        nn.init.normal_(self.lambda_k1, std=0.1)
        nn.init.normal_(self.lambda_k2, std=0.1)

        try:
            from apex.normalization import FusedRMSNorm
            self.ln = FusedRMSNorm(self.head_dim, eps=1e-5, elementwise_affine=True)
        except ImportError:
            self.ln = RMSNorm(self.head_dim, eps=1e-5)

    def forward(self, x):
        b, t, d = x.shape # t: token/sequence length

        q = self.q_linear(x) # (b, t, C) -> (b, t, d); d : head_dim
        k = self.k_linear(x)
        v = self.v_linear(x)

        # Split q and k into two parts
        q1, q2 = torch.chunk(q, 2, dim=-1) # (:,:,d) -> (:,:,d/2)
        k1, k2 = torch.chunk(k, 2, dim=-1)
        
        # Compute Attention Scores
        attn1 = q1 @ k1.transpose(-2, -1) / math.sqrt(self.head_dim / 2)
        attn2 = q2 @ k2.transpose(-2, -1) / math.sqrt(self.head_dim / 2)
        
        # We need to generate a mask as Diff Attn paper trains a decoder only model
        attn_mask = torch.triu(torch.zeros([t, t]).fill_(float("-inf")), diagonal=1)
       
        # Compute Saperate scores
        a1 = f.softmax(attn1+attn_mask / math.sqrt(self.head_dim / 2), dim=-1)
        a2 = f.softmax(attn2+attn_mask / math.sqrt(self.head_dim / 2), dim=-1)
        
        # Compute lambda dynamically
        self.lmbda = torch.exp(torch.sum(self.lambda_q1*self.lambda_k1, dim=-1)) \ 
                    -  torch.exp(torch.sum(self.lambda_q2*self.lambda_k2, dim=-1)) + self.lambda_init
        
        diffattn = (a1 - self.lmbda*a2)@v
        attn = (1 - self.lambda_init)*self.ln(diffattn) 

        return attn

class MultiHead(nn.Module):
    def __init__(self, num_heads, embed_dim, depth):
        super().__init__()
        self.attn_heads = nn.ModuleList([DiffAttn(num_heads, embed_dim, depth) for _ in range(num_heads)])
        self.o_linear = nn.Linear(embed_dim, embed_dim, bias=False) 
    def forward(self, x):
        x = torch.cat([attn_head(x) for attn_head in self.attn_heads], dim=-1)
        out = x*self.o_linear(x)
        return out
