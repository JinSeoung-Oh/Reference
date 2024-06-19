# From https://pub.towardsai.net/rotary-positional-embedding-rope-motivation-and-implementation-ac221926e7df
"""

Positional Embedding in Transformer Models
1. Importance of Positional Embedding
   - Helps transformer models distinguish the order of tokens in a sequence.
   - Without positional embedding, sentences like "My name is Harsh" and "Harsh name is my" would be treated identically.

2. Issues with Absolute Positional Embedding
   - Uses sinusoidal functions to add positional information to token embeddings.
     Problem 1: As sequence length increases, positional embedding vectors become similar, making it harder to distinguish positions, leading to performance degradation in long sequences.
     Problem 2: Encodes absolute positions but fails to capture relative positional information, which is crucial for long-range dependencies and structural patterns in data.

3. Rotary Positional Embedding (RoPE)
   - Motivation
     Introduced to provide positional embedding that incorporates the relative distance between tokens.

   - Mechanism
     Uses the dot product between query and key vectors during self-attention.
     Rotates query and key vectors by an angle proportional to their token positions, making the dot product a function of relative distances.

4. Mathematical Foundation
   - 2D Polar Plane: A vector rotated by 𝜃 is multiplied by 𝑒^(𝑖𝜃)
   - RoPE applies this concept to query and key vectors, rotating them by angles proportional to their positions.
   - Dot product of rotated vectors is proportional to relative distances between tokens.

5. Higher Dimension Extension
   - Embedding vector is broken into chunks of 2.
   - Each chunk is rotated at different frequencies.
     For example, in a 512-dimensional embedding, the first two scalars rotate at frequency 1, the next two at frequency 2, and so on.
   - The rotation matrix affects only pairs of dimensions, allowing for different frequencies across the embedding dimensions.

6. Implementation
   - Detailed the calculation and application of the sinusoidal positional encoding function in PyTorch.
   - Provided an example of the rotation matrix used in RoPE and its application to higher-dimensional embeddings.

7. Key Points
   - Rotational transformation of embeddings ensures that relative positional information is captured.
   - Enhances the model's ability to handle long-range dependencies by focusing on relative positions rather than absolute positions.

8. Implementation and Evaluation
   Absolute Positional Embedding Function (PyTorch)
######################################################################
python
Copy code
def get_sinusoidal_positional_encoding(position, embed_dim, device):
    positional_encoding = torch.zeros(1, embed_dim, device=device)
    pos = torch.tensor([position], dtype=torch.float32, device=device)
    div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32, device=device) * (-math.log(10000.0) / embed_dim))
    positional_encoding[0, 0::2] = torch.sin(pos * div_term)
    positional_encoding[0, 1::2] = torch.cos(pos * div_term)
    return positional_encoding
######################################################################
Explanation:
-Initializes a positional encoding tensor.
-Computes sinusoidal positional encoding for even and odd dimensions.

9. Challenges with Absolute Positional Embedding
   - Similarity of positional vectors for long sequences.
   - Lack of relative positional information.

10. Rotary Positional Embedding (RoPE)
    - Rotates query and key vectors based on token positions.
    - Uses a rotation matrix to apply different frequencies to each chunk of the embedding vector.
    - Ensures the dot product reflects relative distances between tokens.

11. Conclusion
    - RoPE effectively captures relative positional information, addressing the limitations of absolute positional embedding.
    - Enhances the model's ability to manage long-range dependencies and improves performance on tasks with long sequences.
    - The article concludes with the importance of understanding and implementing RoPE in transformer models for better handling of positional information.
"""
# See : https://github.com/huggingface/transformers/blob/main/src/transformers/models/roformer/modeling_roformer.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        #in the equation the power is negative i.e. -(2i-1), so here instead we have computed the same in denominator with +ive power
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
      
    @torch.no_grad()
    def forward(self, position_ids):
        # position_ids: [bs, seq_len]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # inv_freq_expanded: [bs, dim/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()
        # position_ids_expanded: [bs, 1, seq_len]
        device_type = "cuda" if torch.cuda.is_available() else "cpu" 
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # freqs: [bs, dim/2, seq_len] --> transpose --> [bs, seq_len, dim/2]
            cos = freqs.cos()
            # cos: [bs, seq_len, dim/2]
            sin = freqs.sin()
            # sin: [bs, seq_len, dim/2]
        return cos, sin

def apply_rotary_position_embeddings(sin, cos, query_layer, key_layer):
    # sin: [batch_size, sequence_length, embed_size_per_head//2]
    # cos: [batch_size, sequence_length, embed_size_per_head//2]
    # query_layer: [batch_size, sequence_length, embed_size_per_head]
    # key_layer: [batch_size, sequence_length, embed_size_per_head]


    # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    sin_pos = torch.stack([sin, sin], dim=-1).reshape((sin.shape[0], sin.shape[1], sin.shape[2]*2))
    # sin_pos: [batch_size, sequence_length, embed_size_per_head]

    # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
    cos_pos = torch.stack([cos, cos], dim=-1).reshape((cos.shape[0], cos.shape[1], cos.shape[2]*2))
    # cos_pos: [batch_size, sequence_length, embed_size_per_head]

    # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
    rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
        query_layer
    )
    query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos

    # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
    rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
    return query_layer, key_layer




