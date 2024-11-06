## From https://towardsdatascience.com/paper-walkthrough-vision-transformer-vit-c5dcf76f1a7a
## https://towardsdatascience.com/the-ultimate-guide-to-vision-transformers-0a6df32cb248

"""
The Vision Transformer (ViT) represents a significant advancement in computer vision, traditionally dominated by CNNs. 
Prior to ViT's introduction in 2020 by Dosovitskiy et al., 
CNNs were considered superior for vision tasks due to their ability to extract local features through convolutional layers.
However, CNNs require multiple stacked layers to capture global image information. 
ViT addresses this limitation by directly capturing global context from the initial layer.

ViT's architecture is inspired by transformers used in NLP, utilizing only the encoder component. 
It treats image patches as tokens and finds relationships between them. 
The process starts by dividing an image into patches, flattening each, and projecting them into a higher-dimensional space, similar to word embeddings in NLP.

For classification tasks, a class token is prepended to the sequence of projected patches, aggregating information from other patches. 
Positional embeddings are added to preserve spatial information lost during flattening and projection.
The encoded patches pass through a Transformer Encoder block, comprising layer normalization, multi-head attention, and MLP layers, with residual connections applied.
The output from the class token is fed into an MLP head for final classification.

ViT comes in three variants—ViT-B, ViT-L, and ViT-H—differentiated by the number of layers, embedding dimensionality, and other architectural parameters.
These models, including pre-trained versions, have proven to outperform traditional CNNs in various vision tasks.

- CNNs: Focus on local feature extraction through hierarchical layers of convolutions.
- Transformers: Focus on global relationships using self-attention, where each token (like a patch) attends to every other token to capture the context.
Transformers also use Convolution(like, Conv2d) - Some people said "we free from CNN", but I think it is wrong. We still use CNN for data analyze like feature extraction
"""

"""
1. History of ViT:
   The Vision Transformer was introduced by Google Brain in 2021 with the paper "An Image is Worth 16x16 Words."
   It was inspired by the success of Transformers in NLP, first introduced in "Attention is All You Need" in 2017.
   The primary challenge addressed was how to adapt the attention mechanism to images, given that, unlike words, images don’t have a natural tokenization.

2. ViT Architecture:
   Images are divided into patches (e.g., 16x16 pixels), and each patch is treated as a token.  
   A Transformer encoder processes these tokens. However, to classify an image, a special token, known as the CLS token, is introduced. 
   This token is meant to represent the entire image. 
   Positional encoding is used to provide spatial awareness to the tokens, ensuring the model knows the position of each patch.

3. Fine-Tuning at Higher Resolutions:
   Fine-tuning refers to adjusting a pre-trained model for a specific task.
   In the context of Vision Transformers, fine-tuning at a higher resolution means increasing the image resolution while keeping the patch size fixed.
   This increases the number of patches, and positional embeddings are extended via interpolation to accommodate the new token sequence length.

4. Scaling Laws of ViT:
   Vision Transformers excel with larger datasets, but they are more data-hungry compared to CNNs. 
   For small datasets, CNNs tend to perform better due to their built-in inductive biases (like spatial locality and translation invariance).
   As dataset size increases, ViTs begin to outperform CNNs.

5. What Did the Model Learn?:
   ViTs learn embedding filters that resemble those learned by CNNs, despite not using convolutions.
   Positional embeddings show grid-like structures, with the model learning spatial positions on its own.
   The attention mechanism starts by focusing on nearby tokens in early layers and evolves to capture more global relationships in later layers,
   allowing the model to reason over the entire image.

Overall, Vision Transformers bring the benefits of attention-based mechanisms to computer vision, allowing for scalable, 
high-performance models, especially in cases where there is a large amount of data.
"""

import torch
import torch.nn as nn
from torchinfo import summary

BATCH_SIZE   = 1
IMAGE_SIZE   = 224
IN_CHANNELS  = 3

PATCH_SIZE   = 16
NUM_HEADS    = 12
NUM_ENCODERS = 12
EMBED_DIM    = 768
MLP_SIZE     = EMBED_DIM * 4    # 768*4 = 3072

NUM_PATCHES  = (IMAGE_SIZE//PATCH_SIZE) ** 2    # (224//16)**2 = 196

DROPOUT_RATE = 0.1
NUM_CLASSES  = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Patch Flattening & Linear Projection Implementation

class PatcherUnfold(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=PATCH_SIZE, stride=PATCH_SIZE)    
        self.linear_projection = nn.Linear(in_features=IN_CHANNELS*PATCH_SIZE*PATCH_SIZE, 
                                           out_features=EMBED_DIM)    
    def forward(self, x):
        print(f'original\t: {x.size()}')
        
        x = self.unfold(x)
        print(f'after unfold\t: {x.size()}')
        
        x = x.permute(0, 2, 1)    #(1)
        print(f'after permute\t: {x.size()}')
        
        x = self.linear_projection(x)
        print(f'after lin proj\t: {x.size()}')
        
        return x

class PatcherConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=IN_CHANNELS, 
                              out_channels=EMBED_DIM, 
                              kernel_size=PATCH_SIZE, 
                              stride=PATCH_SIZE)
        
        self.flatten = nn.Flatten(start_dim=2)
    
    def forward(self, x):
        print(f'original\t\t: {x.size()}')
        
        x = self.conv(x)    #(1)
        print(f'after conv\t\t: {x.size()}')
        
        x = self.flatten(x)    #(2)
        print(f'after flatten\t\t: {x.size()}')
        
        x = x.permute(0, 2, 1)    #(3)
        print(f'after permute\t\t: {x.size()}')
        
        return x

###### Class Token & Positional Embedding Implementation
class PosEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_token = nn.Parameter(torch.randn(size=(BATCH_SIZE, 1, EMBED_DIM)), 
                                        requires_grad=True)    #(1)
        self.pos_embedding = nn.Parameter(torch.randn(size=(BATCH_SIZE, NUM_PATCHES+1, EMBED_DIM)), 
                                          requires_grad=True)    #(2)
        self.dropout = nn.Dropout(p=DROPOUT_RATE)  #(3)
      
    def forward(self, x):
        class_token = self.class_token
        print(f'class_token dim\t\t: {class_token.size()}')
        
        print(f'before concat\t\t: {x.size()}')
        x = torch.cat([class_token, x], dim=1)    #(1)
        print(f'after concat\t\t: {x.size()}')
        
        x = self.pos_embedding + x    #(2)
        print(f'after pos_embedding\t: {x.size()}')
        
        x = self.dropout(x)    #(3)
        print(f'after dropout\t\t: {x.size()}')
        
        return x

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.norm_0 = nn.LayerNorm(EMBED_DIM)    #(1)
        
        self.multihead_attention = nn.MultiheadAttention(EMBED_DIM,    #(2) 
                                                         num_heads=NUM_HEADS, 
                                                         batch_first=True, 
                                                         dropout=DROPOUT_RATE)
        
        self.norm_1 = nn.LayerNorm(EMBED_DIM)    #(3)
        
        self.mlp = nn.Sequential(    #(4)
            nn.Linear(in_features=EMBED_DIM, out_features=MLP_SIZE),    #(5)
            nn.GELU(), 
            nn.Dropout(p=DROPOUT_RATE), 
            nn.Linear(in_features=MLP_SIZE, out_features=EMBED_DIM),    #(6) 
            nn.Dropout(p=DROPOUT_RATE)
        )
      
    def forward(self, x):
        
        residual = x    #(1)
        print(f'residual dim\t\t: {residual.size()}')
        
        x = self.norm_0(x)    #(2)
        print(f'after norm\t\t: {x.size()}')
        
        x = self.multihead_attention(x, x, x)[0]    #(3)
        print(f'after attention\t\t: {x.size()}')
        
        x = x + residual    #(4)
        print(f'after addition\t\t: {x.size()}')
        
        residual = x    #(5)
        print(f'residual dim\t\t: {residual.size()}')
        
        x = self.norm_1(x)    #(6)
        print(f'after norm\t\t: {x.size()}')
        
        x = self.mlp(x)    #(7)
        print(f'after mlp\t\t: {x.size()}')
        
        x = x + residual    #(8)
        print(f'after addition\t\t: {x.size()}')
        
        return x

class MLPHead(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.linear_0 = nn.Linear(in_features=EMBED_DIM, 
                                  out_features=EMBED_DIM)
        self.gelu = nn.GELU()
        self.linear_1 = nn.Linear(in_features=EMBED_DIM, 
                                  out_features=NUM_CLASSES)    #(1)
        
    def forward(self, x):
        print(f'original\t\t: {x.size()}')
        
        x = self.norm(x)
        print(f'after norm\t\t: {x.size()}')
        
        x = self.linear_0(x)
        print(f'after layer_0 mlp\t: {x.size()}')
        
        x = self.gelu(x)
        print(f'after gelu\t\t: {x.size()}')
        
        x = self.linear_1(x)
        print(f'after layer_1 mlp\t: {x.size()}')
        
        return x

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
    
        #self.patcher = PatcherUnfold()
        self.patcher = PatcherConv()    #(1) 
        self.pos_embedding = PosEmbedding()
        self.transformer_encoders = nn.Sequential(
            *[TransformerEncoder() for _ in range(NUM_ENCODERS)]    #(2)
            )
        self.mlp_head = MLPHead()
    
    def forward(self, x):
        
        x = self.patcher(x)
        x = self.pos_embedding(x)
        x = self.transformer_encoders(x)
        x = x[:, 0]    #(3)
        x = self.mlp_head(x)
        
        return x
