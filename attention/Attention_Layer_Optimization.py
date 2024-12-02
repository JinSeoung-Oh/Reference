### From https://towardsdatascience.com/increasing-transformer-model-efficiency-through-attention-layer-optimization-fefa6f87b1d6

## 1. Baseline: Default Attention
def attn_fn(q, k, v):
    scale = HEAD_DIM ** -0.5
    q = q * scale
    attn = q @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    x = attn @ v
    return x
  
---------------------------------------------------------------------------------------------------
## 2. PyTorch SDPA
# PyTorch's scaled dot-product attention (SDPA) consolidates backend optimizations (FlashAttention-2, Memory-Efficient Attention, 
# C++ Math Attention, and CuDNN). 
# It dynamically selects the most efficient backend.

def set_sdpa_backend(backend):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)

    if backend == 'flash_sdp':
        torch.backends.cuda.enable_flash_sdp(True)
    if backend == 'mem_efficient_sdp':
        torch.backends.cuda.enable_mem_efficient_sdp(True)
    if backend == 'math_sdp':
        torch.backends.cuda.enable_math_sdp(True)
    if backend == 'cudnn_sdp':
        torch.backends.cuda.enable_cudnn_sdp(True)

from torch.nn.functional import scaled_dot_product_attention as sdpa

for backend in ['flash_sdp', 'mem_efficient_sdp', 'math_sdp', 'cudnn_sdp']:
    set_sdpa_backend(backend)
    block_fn = functools.partial(MyAttentionBlock, attn_fn=sdpa)

    print(f'PyTorch SDPA - {backend}')
    train(block_fn, compile=False)
    print(f'Compiled PyTorch SDPA - {backend}')
    train(block_fn, compile=True)

---------------------------------------------------------------------------------------------------
## 3. FlashAttention-3
# FlashAttention-3 improves upon FlashAttention-2 with better memory utilization and fused operations.
from flash_attn_interface import flash_attn_func as fa3

attn_fn = lambda q, k, v: fa3(q, k, v)[0]
block_fn = functools.partial(MyAttentionBlock, attn_fn=attn_fn, format='bshd')

print(f'Flash Attention 3')
train(block_fn)

---------------------------------------------------------------------------------------------------
## 4. Transformer Engine (TE) Attention
# Transformer Engine (TE) provides NVIDIA-specific optimizations for attention layers. It supports backends like Flash, Fused, and Unfused attention.
def set_te_backend(backend):
    os.environ["NVTE_FLASH_ATTN"] = '0'
    os.environ["NVTE_FUSED_ATTN"] = '0'
    os.environ["NVTE_UNFUSED_ATTN"] = '0'

    if backend == 'flash':
        os.environ["NVTE_FLASH_ATTN"] = '1'
    if backend == 'fused':
        os.environ["NVTE_FUSED_ATTN"] = '1'
    if backend == 'unfused':
        os.environ["NVTE_UNFUSED_ATTN"] = '1'

from transformer_engine.pytorch.attention import DotProductAttention

set_te_backend('fused')
attn_fn = DotProductAttention(NUM_HEADS, HEAD_DIM, NUM_HEADS, qkv_format='bshd', attn_mask_type='no_mask')
block_fn = functools.partial(MyAttentionBlock, attn_fn=attn_fn, format='bshd')

print(f'Transformer Engine Attention')
train(block_fn, compile=False)
train(block_fn, compile=True)

---------------------------------------------------------------------------------------------------
## 5. xFormers
# xFormers powers PyTorch's memory-efficient SDPA backend. It offers advanced features when used directly.
from xformers.ops import memory_efficient_attention as mea

block_fn = functools.partial(MyAttentionBlock, attn_fn=mea, format='bshd')

print(f'xFormer Attention')
train(block_fn, compile=False)
train(block_fn, compile=True)

---------------------------------------------------------------------------------------------------
## 6. FlexAttention
# FlexAttention allows for easy customization of the attention computation using a score_mod or block_mask
from torch.nn.attention.flex_attention import flex_attention

def tanh_softcap(score, b, h, q_idx, kv_idx):
    return 30 * torch.tanh(score / 30)

flex_fn = functools.partial(flex_attention, score_mod=tanh_softcap)
block_fn = functools.partial(MyAttentionBlock, attn_fn=flex_fn)

print(f'Flex Attention with Softcap')
train(block_fn, compile=True)

def mask_mod(b, h, q_idx, kv_idx):
    return torch.abs(q_idx - kv_idx) < 5

block_mask = create_block_mask(mask_mod, None, None, SEQ_LEN, SEQ_LEN)
block_fn = functools.partial(flex_attention, block_mask=block_mask)

print(f'Flex Attention with Mask')
train(block_fn, compile=True)

---------------------------------------------------------------------------------------------------
"""
7. Summary of Results
   Kernel	Uncompiled Step Time (ms)	Compiled Step Time (ms)
   Default Attention	370	242
   PyTorch SDPA (Flash)	247	203
   FlashAttention-3	240	N/A
   Transformer Engine	243	204
   xFormers	246	203
   FlexAttention	~258	210

8. Conclusion
   - FlashAttention-3 performs best in eager execution mode.
   - PyTorch SDPA and Transformer Engine excel in compiled mode.
   - FlexAttention is ideal for custom attention modifications without sacrificing performance.
"""
