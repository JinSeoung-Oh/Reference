### From https://towardsdatascience.com/the-rise-of-sparse-mixtures-of-experts-switch-transformers-39cf3671f8a0
"""
1. Sparse Mixtures of Experts (MoE) Recap
   Sparse Mixture of Experts (MoE) is a key technique in the latest generation of Large Language Models (LLMs), such as OpenAI’s GPT-4 and Mistral AI’s Mixtral-8x7. 
   The core idea of sparse MoE is to increase the model’s capacity (number of parameters) without increasing its computational cost proportionally. 
   In an ideal scenario, you can scale the number of parameters to extremely large values while maintaining roughly O(1) computational cost per token.

   However, making sparse MoE work effectively in practice involves careful design decisions — particularly around how tokens (input elements) are routed to different experts 
   and how to balance efficiency with model quality.

2. Hard Routing vs. Soft Routing
   Traditionally, Mixture of Experts models used “soft routing,” where the gate (or router) assigns a probability to each expert, 
   and the final output is a weighted combination of all experts’ outputs. This is computationally expensive since even low-weight experts still get executed.

   In “hard routing,” only the top-1 (or top-k) expert is selected and executed for each token. This drastically reduces computational overhead 
   because you avoid running many experts with negligible weights. The Switch Transformer (Fedus et al., 2022) shows that k=1 hard routing actually works well, 
   contrary to earlier assumptions.

   -1. Mathematical Formulation:
       -a. Soft Routing:
           𝑦 = (∑ 𝑖=1 to 𝐸)𝐺(𝑥)_𝑖 𝐸_𝑖(𝑥)
           where 𝐺(𝑥) is a softmax distribution over experts and 𝐸_𝑖 is the i-th expert.

       -b. Hard Routing:
           𝑦 ≈ 𝐸_𝑗(𝑥) where 𝑗=arg max_𝑖 𝐺(𝑥)_𝑖
​           Here, 𝐺(𝑥)_𝑖 = exp(𝑥𝑊_𝑖) / ∑_𝑘 exp(𝑥𝑊_𝑘) and 𝑊_𝑖 are learnable parameters of the gate.

3. The Switch Transformer Architecture
   A standard Transformer block typically consists of:
   -1. A Multi-Head Self-Attention layer. 
   -2. A single dense Feed-Forward Network (FFN) layer.

   In a Switch Transformer block, the single FFN is replaced by a bank of FFN “experts.” A router determines which expert is chosen for each token.
   Because only one expert processes a given token, the computational cost remains similar to a dense Transformer, 
   but the model’s total parameter count is much larger (since it includes many FFNs).

   -1. Dense Transformer Block:
       [Input] --> [Self-Attention] --> [FFN] --> [Output]
   -2. Switch Transformer Block:
       [Input] 
          --> [Self-Attention] 
          --> [Router decides expert i for each token]
          --> [Expert i (FFN_i)]
          --> [Output]
       
       Though we’ve added multiple FFNs, only one expert runs per token, keeping computation roughly constant while increasing capacity.

4. Token Routing Dynamics and Capacity Factor
   Routing tokens to experts in a distributed setting (where each expert may be on a different machine) introduces the concept of “capacity.” 
   Each expert can handle only a certain number of tokens at a time due to memory and computation constraints. The capacity factor 𝑓 controls how many tokens per expert 
   are allowed.

   -1. Capacity:
       capacity = 𝑓×(𝑇/𝐸)
       where 𝑇 is the total number of tokens, 𝐸 is the number of experts, and 𝑓 is the capacity factor.
       - If 𝑓 is too small, some tokens must be dropped (losing training signal).
       - If 𝑓 is too large, we waste resources on padding.
       Empirically, the Switch Transformer authors found optimizing for machine utilization (less padding, even if that means dropping some tokens) leads to better model 
       performance.

5. Scaling Properties of the Switch Transformer
   By using hard routing and sparse MoE layers, the Switch Transformer achieves remarkable scaling efficiency. 
   Adding more experts increases total capacity (parameters) without increasing the FLOPs per token, because the gating ensures only one expert runs per token.

   Fedus et al. (2022) report a 7X speed-up in training compared to a dense T5 model at the same quality level. 
   This is essentially getting the same performance as a large dense model but trained much faster under the same computational budget.

"""
### 6. Code Examples
## Note: The original text does not provide any code. Below we provide illustrative pseudo-code and example code snippets inspired by open-source implementations 
## (e.g., Google/DeepMind research code, PyTorch implementations from Fairseq, or JAX/TensorFlow implementations from T5X and Switch Transformer repositories). 
## These snippets are conceptual and show how one might implement routing and MoE layers.

### Pseudo-code for the Router (Hard Routing)
import torch
import torch.nn.functional as F

def hard_router(x, W):
    # x: [batch_size, seq_length, hidden_dim]
    # W: [hidden_dim, num_experts] router weight matrix
    # returns expert_indices: [batch_size, seq_length]

    # Compute logits: how suitable each expert is for each token
    logits = torch.einsum('bsh,he->bse', x, W)  # [batch, seq, experts]
    # Apply softmax to get probabilities (or just argmax directly)
    # For hard routing, we just pick the top-1 expert:
    expert_indices = torch.argmax(logits, dim=-1)  # [batch, seq]
    return expert_indices

## The code above shows a simplified router: it projects tokens x onto a space defined by the router weights W, yielding a logit per expert. 
## Then, for hard routing, we choose the top expert per token with argmax.

### MoE Layer With Hard Routing
class MoELayer(torch.nn.Module):
    def __init__(self, hidden_dim, expert_ffn_dim, num_experts):
        super().__init__()
        # Create multiple FFNs (experts)
        self.num_experts = num_experts
        self.experts = torch.nn.ModuleList([
            FFN(hidden_dim, expert_ffn_dim) for _ in range(num_experts)
        ])
        # Router weights
        self.router_w = torch.nn.Parameter(torch.empty(hidden_dim, num_experts))
        torch.nn.init.xavier_uniform_(self.router_w)

    def forward(self, x):
        # x: [batch, seq, hidden_dim]

        # Determine expert assignments
        expert_indices = hard_router(x, self.router_w)

        # Group tokens by expert
        # For efficiency, we typically perform a gather operation that
        # collects tokens going to the same expert, process them in a single
        # batched call, then scatter them back.
        batch, seq, hdim = x.shape
        # Flatten for indexing
        x_flat = x.reshape(batch * seq, hdim)
        expert_flat = expert_indices.reshape(batch * seq)

        # Suppose we handle routing per expert
        outputs = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            mask = (expert_flat == expert_id)
            selected_tokens = x_flat[mask]  # tokens assigned to this expert
            if selected_tokens.shape[0] > 0:
                # Run through the expert FFN
                out = self.experts[expert_id](selected_tokens)
                # Place results back
                outputs[mask] = out

        # Reshape back
        outputs = outputs.reshape(batch, seq, hdim)
        return outputs

### In the pseudo-code above:
# We first run the router to find the best expert for each token.
# We then group tokens by the assigned expert, run them through that expert’s FFN, and finally restore the original order.
# Real implementations also handle padding or dropping tokens if capacity is exceeded.

--------------------------------------------------------------------------------------------------------------------------
### Implementing Capacity Constraints
def capacity_constrained_routing(x, W, capacity_factor):
    batch, seq, hdim = x.shape
    num_experts = W.shape[1]
    total_tokens = batch * seq
    capacity = int(capacity_factor * (total_tokens / num_experts))

    # Compute logits and argmax
    logits = torch.einsum('bsh,he->bse', x, W)
    expert_indices = torch.argmax(logits, dim=-1)  # [batch, seq]

    # Count how many tokens per expert
    token_counts = torch.bincount(expert_indices.view(-1), minlength=num_experts)

    # If an expert is over capacity, drop excess tokens
    # This is a simplified approach; real implementations may sort by score.
    for expert_id in range(num_experts):
        if token_counts[expert_id] > capacity:
            # Identify tokens assigned to this expert
            token_positions = (expert_indices == expert_id).nonzero(as_tuple=True)
            # Drop excess tokens (e.g., the last ones)
            # Real code might pick tokens with the highest router logits to keep.
            excess = token_counts[expert_id] - capacity
            drop_positions = token_positions[0][-excess:]
            # Mark these dropped tokens with a special expert_id = -1
            expert_indices[token_positions[0][-excess:]] = -1

    return expert_indices

# This snippet shows a conceptual method to enforce capacity constraints. Excess tokens assigned to a particular expert are dropped. 
# In practice, to minimize performance loss, the router might rank tokens by the gating score and drop the lowest-scoring tokens first.

-------------------------------------------------------------------------------------------------------------------------------------
""" 
7. Key Takeaways from the Switch Transformer Paper
   -1. Hard Routing Works: Contrary to earlier assumptions, using k=1 (hard routing) is sufficient to achieve excellent performance and scale effectively.

   -2. O(1) Scaling in Theory: By increasing experts, we increase model capacity without increasing per-token FLOPs. 
                               The top-1 routing keeps computation constant per token.

   -3. Capacity Factor is Crucial: Adjusting the capacity factor helps balance resource utilization and performance. 
                                   The authors found that it’s typically better to choose a capacity factor that fully utilizes machine resources, 
                                   even if it means dropping some tokens.

   -4. 7X Speed-up: By integrating sparse MoE layers into a T5-like architecture, 
                    the Switch Transformer achieves a 7-fold training speed improvement at the same quality level as a dense baseline.

8. Conclusion
   The Switch Transformer demonstrates how sparse MoE and hard routing enable massive scaling of model capacity without linear growth in computation. 
   The code snippets above illustrate how such a routing mechanism might be implemented in practice. 
   While simplified, they capture the essence of how tokens are assigned to experts, how capacity constraints are handled, 
   and how the computations remain constant despite growing parameter counts.

   Sparse MoE technology, as showcased by Switch Transformers, GPT-4, and Mistral’s Mixtral-8x7, is likely just the beginning.
   With further refinements and research, we’ll continue to see even more efficient and capable models across various domains, 
   fueled by the concepts and implementations laid out in this line of research.
"""
