# From https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch

## The self attention layers in Sparse mixture of experts models are the same as in regular transformer models
## Have to check all archtecture image from https://huggingface.co/blog/AviSoori1x/makemoe-from-scratch, below top-k router class

# Why sparse MoE?
# Sparse MoE (희소 Mixture of Experts): 희소 MoE는 MoE 구조를 희소하게 만드는 방법을 의미. 
# 이는 전체 전문가 중에서 일부만이 특정 입력 또는 상황에서 활성화되고, 나머지는 비활성화되는 형태
# 이렇게 함으로써 모델의 파라미터를 효과적으로 관리하고, 계산 리소스를 효율적으로 사용할 수 있음

import torch
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import torch.nn.init as init

torch.manual_seed(1337)
B,T,C = 4,8,32 # B = batch, T = time, C = channels
x = torch.randn(B,T,C)

## sigle head perform self-attention
head_size=16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)

k = key(x)   # (B,T,16)
q = qurey(x) # (B,T,16)
wei =  q @ k.transpose(-2, -1) # (B, T, 16) @ (B, 16, T) ---> (B, T, T)
tril = torch.tril(torch.ones(T,T)) # torch.tril <-- 하삼각 생성, https://velog.io/@sang9/%ED%8C%8C%EC%9D%B4%ED%86%A0%EC%B9%98-%EB%AA%85%EB%A0%B9%EC%96%B41-torch.triu

wei = wei.masked_fill(tril==0, float('-inf'))
# tril == 0: tril과 동일한 모양의 이진 마스크를 생성. 여기서 삼각 행렬의 하단 부분(대각선을 제외한)은 True로 설정되고, 나머지는 False로 설정
# wei.masked_fill(..., float('-inf')): 이진 마스크에서 True인 위치에 해당하는 wei의 요소를 -inf로 교체
# 어텐션 메커니즘의 맥락에서 특정 값을 -inf로 설정하면 softmax를 사용하여 어텐션 스코어를 계산할 때 해당 위치의 영향이 무시되도록 할 수 있음

wei = F.softmax(wei, dim=-1) # B,T,T
v = value(x) # B,T,H
out = wei @ v # (B,T,T) @ (B,T,H) -> (B,T,H)

## Causal scaled dot product self-Attention Head
n_embd = 64
n_head = 4
n_layer = 4
head_size = 16
dropout=0.1

class Head(nn.Module):
  """ one head of self-attention """
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch,ones(block_size, block_size)))
    self.dropout = nn,Dropout(dropout)

  def forward(self,x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T)
    wei = F.softmax(wei, dim=-1) # (B,T,T)
    v = self.value(x)
    out = wei @ v
    return out

## Multi-head self attention
class MultiHeadAttention(nn.Module):
  def __ini__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]) # create num_heads Head classes
    self.proj = nn.Linear(n_embd, n_embd)
    self.dropout = nn.Dropout(dropout)
  def forward(self,x):
    out = torch.cat([h(x) for h in self.heads], dim = -1)
    out = self.dropout(self.proj(out))
    return out

## Expert "Module" not "model"
class Expert(nn.Module):
  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd),
      nn.ReLU(),
      nn.Linear(4*n_embd, n_embd),
      nn.Dropout(dropout),
    )
  def forward(self,x):
    return self.net(x)

## Router : Top-k gating
class TopkRouter(nn.Module):
  def __init__(self, n_embed, num_experts, top_k):
    super(TopkRouter, self).__init__()
    self.top_k = top_k
    self.linear = nn.Linear(n_embed, num_experts)
  def forward(self, mh_out):
    # mh_ouput is the output tensor from multihead self attention block
    logits = self.linear(mh_output)
    top_k_logits, indices = logits.topk(self.top_k, dim=-1)
    zeros = torch.full_like(logits, float('-inf'))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_output = F.softmax(sparse_logits, dim=-1)
    return router_output, indices

## Useage example
num_experts = 4
top_k = 2
n_embd = 32

mh_output = torch.randn(2, 4, n_embd)  # Example input
top_k_gate = TopkRouter(n_embd, num_experts, top_k)
gating_output, indices = top_k_gate(mh_output)
gating_output.shape, gating_output, indices

## Noisy top-k Gating
class NoisyTopkRouter(nn.Module):
  def __init__(self, n_embed, num_experts, top_k):
    super(NoisyTopkRouter, self).__init__()
    self.top_k = top_k
    self.topkroute_linear = nn.Linear(n_embed, num_experts)
    self.noise_linear = nn.Linear(n_embed, num_experts)
  
  def forward(self, mh_output):
    logits = self.topkroute_linear(mh_output)
    noise_logits = self.noise_linear(mh_output)
    noise = torch.randn_like(logits)*F.softplus(noise_logits)
    noisy_logits = logits + noise
    top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
    zeros = torch.full_like(noisy_logits, float('-inf'))
    sparse_logits = zeros.scatter(-1, indices, top_k_logits)
    router_output = F.softmax(sparsee_logits, dim=-1)
    return router_output, indices

## Creating a sparse Mixture of Experts module
class SparseMoE(nn.Module):
  def __init__(self, n_embed, num_experts, top_k):
    super(SparseMoE, self).__init__()
    self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
    self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
    self.top_k = top_k
    
  def forward(self, x):
    gating_output, indices = self.router(x)
    final_output = torch.zeros_like(x)
    flat_x = x.view(-1, x.size(-1))
    flat_gating_output = gating_output.view(-1, gating_output.size(-1))
    for i, expert in enumerate(self.experts):
      # Create a mask for the inputs where the current expert is in top-k
      expert_mask = (indices == i).any(dim=-1)
      flat_mask = expert_mask.view(-1)
      if flat_mask.any():
        expert_input = flat_x[flat_mask]
        expert_output = expert(expert_input)
        gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
        weighted_output = expert_ouptput * gating_scores
        final_output.masked_scatter_(expert_mask.unsqueeze(-1), weighted_output)

    return final_output.view_as(x)

## All code
 class Block(nn.Module):
    """ Mixture of Experts Transformer block: communication followed by computation (multi-head self attention + SparseMoE) """

    def __init__(self, n_embed, n_head, num_experts, top_k):
        # n_embed: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.smoe = SparseMoE(n_embed, num_experts, top_k)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.smoe(self.ln2(x))
        return x

 class SparseMoELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, num_experts=num_experts,top_k=top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Initialization is important for efficient training of deep neural nets. 
# Kaiming He initialization is used here because of presence of ReLU activations in the experts. 
# Feel free to experiment with Glorot initialization which is more commonly used in transformers

def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)): 
        init.kaiming_normal_(m.weight)

model = SparseMoELanguageModel()
model.apply(kaiming_init_weights)

#Using MLFlow
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#mlflow.set_experiment("makeMoE")
with mlflow.start_run():
    #If you use mlflow.autolog() this will be automatically logged. I chose to explicitly log here for completeness
    params = {"batch_size": batch_size , "block_size" : block_size, "max_iters": max_iters, "eval_interval": eval_interval,
              "learning_rate": learning_rate, "device": device, "eval_iters": eval_iters, "dropout" : dropout, "num_experts": num_experts, "top_k": top_k }
    mlflow.log_params(params)
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            metrics = {"train_loss": losses['train'], "val_loss": losses['val']}
            mlflow.log_metrics(metrics, step=iter)


        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
