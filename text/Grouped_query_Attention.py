"""
From https://medium.com/towards-artificial-intelligence/grouped-query-attention-gqa-explained-5f3dbbfe013b

## What is GQA
   Grouped-query Attention (GQA) is a mechanism used in large language models, such as Llama 2, 
   to address the challenge of memory cost associated with autoregressive decoding and multi-head attention (MHA). 
   The primary goal of GQA is to balance the trade-off between memory efficiency and inference quality.

1. Memory Efficiency: 
   GQA is designed to reduce memory requirements during autoregressive decoding by modifying the attention mechanism. 
   Specifically, it focuses on the key and value cache associated with attention computation.

2. Reduction of Key-Value Cache: 
   GQA achieves memory efficiency by utilizing a reduced number of key-value heads compared to the total number of query heads. 
   This reduction allows for a smaller key-value (kv) cache, 
   which is crucial for handling larger context windows or batch sizes without a significant increase in memory cost.

3. Interpolation of Mechanisms:
   GQA is described as an interpolation between two existing mechanisms—Multi-Query Attention (MQA) and Multi-Head Attention (MHA). 
   While MQA uses a single key-value head for multiple queries to save memory, GQA extends this idea by incorporating multiple key-value heads 
   but with fewer than the total number of query heads, striking a balance between memory efficiency and quality.

4. Implementation in LLMs:
   The article provides a code snippet that demonstrates how GQA is implemented in the Attention module of the Llama 2 large language model.
   The implementation includes the use of ColumnParallelLinear and RowParallelLinear for model parallelism.

5. Repeat_kv Function: 
   GQA uses a function called repeat_kv to duplicate keys and values, aligning them with the number of query heads. 
   This step is crucial for ensuring that the dimensions match during the matrix multiplication (GEMM) subroutine involved in attention computation.

In summary, GQA is a memory-efficient attention mechanism that reduces the memory footprint associated with attention computation in autoregressive decoding. 
It achieves this by strategically adjusting the number of key-value heads while maintaining a balance between memory efficiency and inference quality.

## Grouped-query attention (GQA) mechanism in the context of large language models (LLMs) like Llama 2.
1. Autoregressive Decoding and Memory Cost:
   Autoregressive decoding involves caching keys and values from previous tokens to speed up attention computation. 
   However, as the context window or batch size increases, the memory cost associated with the key-value cache (kv cache) 
   in the multi-head attention (MHA) model significantly increases.

2. Multi-Query Attention (MQA): 
   MQA is introduced as a mechanism that uses only a single key-value head for multiple queries, aiming to save memory and speed up decoder inference. 
   However, it may lead to a decrease in quality.

3. Grouped-query Attention (GQA): 
   GQA is presented as an interpolation of MQA and MHA. 
   It aims to achieve a quality similar to MHA while maintaining comparable speed to MQA. 
   GQA is designed to enhance inference quality by using multiple keys and values heads with fewer than the total number of query heads.

4. Implementation in LLM (Llama 2): 
   The provided Python code snippet shows the implementation of the Attention module in Llama 2, incorporating the GQA mechanism. 
   The attention module includes the use of ColumnParallelLinear and RowParallelLinear for model parallelism.

5. Repeat_kv Function: 
   The repeat_kv function is defined to duplicate keys and values, aligning them with the number of query heads. 
   This is essential for matching the dimensions required for the matrix multiplication (GEMM) subroutine during the attention computation.

6. Conclusion: 
   The conclusion emphasizes that both GQA and MQA aim to reduce the computational load, primarily by decreasing the need for storing a large kv cache. 
   This reduction in memory usage allows LLM servers to handle more requests, larger batch sizes, and increased throughput.

Overall, the text provides insights into the motivations behind GQA, its implementation in LLMs, 
and its potential benefits in terms of memory efficiency and throughput in large-scale language processing tasks.

## What is Main difference GQA and MHA based on this article(https://medium.com/towards-artificial-intelligence/grouped-query-attention-gqa-explained-5f3dbbfe013b)
The main difference between Grouped-query Attention (GQA) and Multi-Head Attention (MHA), 
as described in the provided article, lies in their approach to handling attention computation and memory efficiency during autoregressive decoding.

1. Memory Efficiency Strategy:
   -1. Multi-Head Attention (MHA): 
       MHA uses multiple query, key, and value heads for attention computation. Each query attends to all keys, 
       and the attention scores are computed in parallel for each head. This approach leads to a larger key-value (kv) cache, 
       which can become memory-intensive, especially with increasing context windows or batch sizes.
   -2. Grouped-query Attention (GQA): 
       GQA introduces a strategy to reduce memory requirements by using fewer key-value heads compared to the total number of query heads. 
       This reduction allows for a smaller kv cache, addressing the memory cost associated with MHA.

2. Interpolation Between Mechanisms:
   -1. MHA: 
       MHA involves parallel computation across multiple heads without reducing the number of key-value heads. 
       It is a widely used attention mechanism in many models, including transformers.
   -2. GQA: 
       GQA is described as an interpolation between MHA and Multi-Query Attention (MQA). 
       While MQA uses a single key-value head for multiple queries to save memory, GQA extends this idea 
       by incorporating multiple key-value heads but with fewer than the total number of query heads. 
       GQA strikes a balance between the memory efficiency of MQA and the parallel computation of MHA.

3. Implementation in LLMs:
   -1. MHA: 
       The article doesn't explicitly provide MHA implementation details, but MHA typically involves parallel computations across multiple heads, 
       each attending to the entire sequence.
  -2. GQA: 
      The article includes a code snippet illustrating how GQA is implemented in the Attention module of the Llama 2 model. 
      GQA involves the use of ColumnParallelLinear and RowParallelLinear for model parallelism, along with a function called repeat_kv to duplicate keys and values.

4. Reduced Key-Value Cache:
   -1. MHA: 
       MHA uses the full set of key-value heads, leading to a larger kv cache.
   -2. GQA: 
       GQA strategically reduces the number of key-value heads, resulting in a smaller kv cache. 
       This reduction is achieved by utilizing an interpolation strategy that balances memory efficiency and attention quality.

In summary, GQA is introduced as a mechanism to improve memory efficiency during autoregressive decoding, 
specifically targeting the key-value cache associated with attention computation. 
It achieves this by reducing the number of key-value heads while maintaining a balance between memory efficiency and the quality of attention computations, 
making it a suitable choice for large language models like Llama 2.

"""

# From https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a

import torch
from einops import rearrange,einsum
import torch.nn.functional as F

# shapes: (batch_size, seq_len, num_heads, head_dim)
query = torch.randn(1, 256, 8, 64)
key = torch.randn(1, 256, 2, 64)
value = torch.randn(1, 256, 2, 64)

num_head_groups = query.shape[2] // key.shape[2]
scale = query.size(-1) ** 0.5

#Swap seq_len with num_head to accelerate computations
#Have to check below line
query = rearrange(query, "b n h d -> b h n d")
key = rearrange(key, "b s h d -> b h s d")
value = rearrange(value, "b s h d -> b h s d")

#Split query num_heads in groups by introducing additioanl 'g' dimension
query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)

# g stands for the number of groups
# h stands for the hidden dim
# n and s are equal and stands for sequence length
 
scores = einsum(query, key, "b g h n d, b h s d -> b h n s")
attention = F.softmax(scores/scale, dim=1)
out = einsum(attention, value, "b h n s, b h s d -> b h n d")
out = rearrange(out, "b h n d -> b n h d")
