### From https://ai.gopubby.com/beyond-tokens-how-byte-latent-transformers-blt-are-solving-the-10b-language-model-crisis-da0c9788fd47

"""
1. Introduction to BLT
   -1. Key Idea: The Byte Latent Transformer (BLT) eliminates traditional tokenization by working directly at the byte-level. 
                 Instead of predefined tokens (like words or subwords), BLT treats all text as sequences of bytes, enabling more efficient processing, 
                 better multilingual handling, and dynamic segmentation. This leads to:

                 -a. Reduced Overheads: No large token vocabularies.
                 -b. Better Multilingual Support: Uniform handling of all languages and scripts.
                 -c. Robustness to Noise: Byte-level modeling mitigates issues like typos and unfamiliar characters.
                 -d. Improved Efficiency: Lower memory footprint, reduced latency, and simpler scaling.

   -2. Potential Gains:
       -a. Up to 50% fewer inference FLOPs.
       -b. 60% reduced GPU memory footprint.
       -c. Faster training and deployment with no complex tokenization rules.
"""

## Example Code: Traditional Tokenization vs. Byte-Level Representation
#** Traditional Tokenization Example (not part of BLT, shown for contrast):

from transformers import GPT2Tokenizer

# Tokenizing with a traditional approach
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
text = "Hello, world!"
tokens = tokenizer.encode(text)
print(tokens)  # e.g. [15496, 11, 995, 0]

#** Byte-Level Representation (BLT approach):
# No tokenization, just bytes
text = "Hello, world!"
byte_sequence = list(text.encode('utf-8'))
print(byte_sequence)  # e.g. [72, 101, 108, 108, 111, 44, 32, 119, 111, 114, 108, 100, 33]

"""
2. A Simplified BLT-like Model Architecture in PyTorch
   Below is a conceptual implementation of a byte-level transformer model. 
   This example demonstrates how one might implement a BLT-like approach at a high level. 
   It uses standard TransformerEncoder layers and a simple byte embedding layer instead of a complex token vocabulary.
"""
import torch
import torch.nn as nn
import math

class ByteLatentTransformer(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_layers=6, ff_dim=2048, max_seq_len=1024, byte_size=256):
        super().__init__()
        # Instead of a token embedding, we use a byte embedding directly:
        self.byte_embedding = nn.Embedding(byte_size, embed_dim)

        # Positional encoding to retain sequence information
        self.positional_encoding = self._create_positional_encoding(embed_dim, max_seq_len)

        # Define the Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output layer predicts next byte or another supervised target
        self.output_layer = nn.Linear(embed_dim, byte_size)

    def forward(self, byte_seq):
        # byte_seq: (batch_size, seq_length)
        embeddings = self.byte_embedding(byte_seq) + self.positional_encoding[:byte_seq.size(1), :]
        # Transform to shape (seq_length, batch_size, embed_dim) for PyTorch's Transformer
        embeddings = embeddings.transpose(0, 1)

        # Pass through transformer
        encoded = self.transformer(embeddings)

        # Compute logits for next-byte prediction
        logits = self.output_layer(encoded)  # (seq_length, batch_size, byte_size)
        return logits.transpose(0, 1)

    def _create_positional_encoding(self, embed_dim, max_len):
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, embed_dim)
"""
3. Key Points in the Code:
   -a. byte_embedding: Maps each byte (0–255) to a vector, eliminating token vocabularies.
   -b. positional_encoding: Provides positional context.
   -c. transformer: Uses standard Transformer layers to model byte sequences.
   -d. output_layer: Predicts the next byte distribution.
"""

## Training the BLT Model (Conceptual)
#** Training involves feeding raw bytes, no special tokenization needed:

# pseudo training loop
model = ByteLatentTransformer()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for byte_seq, next_byte in train_loader:
        optimizer.zero_grad()
        logits = model(byte_seq)  # (batch, seq_length, byte_size)
        # Shift for next-byte prediction
        loss = criterion(logits[:, :-1].reshape(-1, 256), next_byte[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

# Note: This code is illustrative and omits data loading and other training details. A real BLT system might add dynamic patch segmentation, 
# advanced attention patterns, or integrate specialized components for efficiency.

"""
4. Conclusion
   The Byte Latent Transformer represents a significant step forward by eliminating the complexity of tokenization. 
   Instead of wrestling with large vocabularies and arbitrary segmentation rules, BLT handles raw bytes directly. This approach provides:

   -a. Better Multilingual Support: Uniform handling of all languages and scripts.
   -b. Robustness to Noise and Rare Tokens: Model never encounters an "unknown token."
   -c. Efficiency Gains: Lower training costs, reduced memory footprint, and faster deployment.

   Adopting BLT will shape the future of NLP, making models more scalable, flexible, and accessible for developers who want to handle a wide range of languages 
   and domains without tokenization headaches.
"""
