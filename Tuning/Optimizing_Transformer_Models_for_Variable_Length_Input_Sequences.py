### From https://towardsdatascience.com/optimizing-transformer-models-for-variable-length-input-sequences-19fb88fddf71

"""
1. Introduction
   Generative AI models, built on Transformer architecture, have significant computational demands due to the attention mechanism's 
   high complexity. Optimizing attention mechanisms is critical, especially when working with variable-length input sequences like documents,
   code, or time-series data. This post explores methods to optimize handling variable-length sequences, 
   comparing padding-based and padding-free approaches while integrating advanced tools like FlashAttention2 and PyTorch NestedTensors.

2. Challenge: Variable-Length Input
   -1. Batching: Efficient computation often relies on batching input sequences. However, tensors in a batch must have the same shape.
   -2. Padding: The common solution is padding sequences to the same length, but this wastes GPU resources.
   -3. Concatenation: An alternative is concatenating sequences and using attention masks to ensure tokens attend only to their 
                      respective sequence. However, naive implementations face 𝑂(𝑁^2) complexity.

3. Proposed Optimizations
   -1. PyTorch SDPA (Scaled Dot Product Attention) with Padding
   -2. Dynamic Padding to the Longest Sequence
   -3. SDPA with PyTorch NestedTensors
   -4. FlashAttention2 for Variable-Length Input
   -5. xFormers' Memory-Efficient Attention
   -6. HuggingFace Model Integration with FlashAttention2
"""

### Transformer Block
# general imports
import time, functools

# torch imports
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

# Define Transformer settings
BATCH_SIZE = 32
NUM_HEADS = 16
HEAD_DIM = 64
DIM = NUM_HEADS * HEAD_DIM
DEPTH = 24
NUM_TOKENS = 1024
MAX_SEQ_LEN = 1024
PAD_ID = 0
DEVICE = 'cuda'

class MyAttentionBlock(nn.Module):
    def __init__(
            self,
            attn_fn,
            dim,
            num_heads,
            format=None,
            **kwargs
    ):
        super().__init__()
        self.attn_fn = attn_fn
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.norm1 = nn.LayerNorm(dim, bias=False)
        self.norm2 = nn.LayerNorm(dim, bias=False)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # mlp layers
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

        self.permute = functools.partial(torch.transpose, dim0=1, dim1=2)
        if format == 'bshd':
            self.permute = nn.Identity()

    def mlp(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    def reshape_and_permute(self,x, batch_size):
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return self.permute(x)

    def forward(self, x_in, attn_mask=None):
        batch_size = x_in.size(0)
        x = self.norm1(x_in)
        qkv = self.qkv(x)

        # rather than first reformatting and then splitting the input
        # state, we first split and then reformat q, k, v in order to
        # support PyTorch Nested Tensors
        q, k, v = qkv.chunk(3, -1)
        q = self.reshape_and_permute(q, batch_size)
        k = self.reshape_and_permute(k, batch_size)
        v = self.reshape_and_permute(v, batch_size)
        
        # call the attn_fn with the input attn_mask
        x = self.attn_fn(q, k, v, attn_mask=attn_mask)

        # reformat output
        x = self.permute(x).reshape(batch_size, -1, self.dim)
        x = self.proj(x)
        x = x + x_in
        x = x + self.mlp(self.norm2(x))
        return x
      
-------------------------------------------------------------------------------------------------------
### Transformer Decoder Model
class MyDecoder(nn.Module):
    def __init__(
            self,
            block_fn,
            num_tokens,
            dim,
            num_heads,
            num_layers,
            max_seq_len,
            pad_idx=None
    ):
        super().__init__()
        self.num_heads = num_heads
        self.pad_idx = pad_idx
        self.embedding = nn.Embedding(num_tokens, dim, padding_idx=pad_idx)
        self.positional_embedding = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            block_fn(
                dim=dim,
                num_heads=num_heads
            )
            for _ in range(num_layers)])
        self.output = nn.Linear(dim, num_tokens)

    def embed_tokens(self, input_ids, position_ids=None):
        x = self.embedding(input_ids)
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1],
                                        device=x.device)
        x = x + self.positional_embedding(position_ids)
        return x

    def forward(self, input_ids, position_ids=None, attn_mask=None):
        # Embed tokens and add positional encoding
        x = self.embed_tokens(input_ids, position_ids)
        if self.pad_idx is not None:
            assert attn_mask is None
            # create a padding mask - we assume boolean masking
            attn_mask = (input_ids != self.pad_idx)
            attn_mask = attn_mask.view(BATCH_SIZE, 1, 1, -1) \
                .expand(-1, self.num_heads, -1, -1)

        for b in self.blocks:
            x = b(x, attn_mask)

        logits = self.output(x)
        return logits

-------------------------------------------------------------------------------------------------------
### Variable Length Sequence Input
# Use random data
class FakeDataset(Dataset):
    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        length = torch.randint(1, MAX_SEQ_LEN, (1,))
        sequence = torch.randint(1, NUM_TOKENS, (length + 1,))
        inputs = sequence[:-1]
        targets = sequence[1:]
        return inputs, targets

def pad_sequence(sequence, length, pad_val):
    return torch.nn.functional.pad(
        sequence,
        (0, length - sequence.shape[0]),
        value=pad_val
    )

def collate_with_padding(batch):
    padded_inputs = []
    padded_targets = []
    for b in batch:
        padded_inputs.append(pad_sequence(b[0], MAX_SEQ_LEN, PAD_ID))
        padded_targets.append(pad_sequence(b[1], MAX_SEQ_LEN, PAD_ID))
    padded_inputs = torch.stack(padded_inputs, dim=0)
    padded_targets = torch.stack(padded_targets, dim=0)
    return {
        'inputs': padded_inputs,
        'targets': padded_targets
    }

def data_to_device(data, device):
    if isinstance(data, dict):
        return {
            key: data_to_device(val,device)
            for key, val in data.items()
        }
    elif isinstance(data, (list, tuple)):
        return type(data)(
            data_to_device(val, device) for val in data
        )
    elif isinstance(data, torch.Tensor):
        return data.to(device=device, non_blocking=True)
    else:
        return data.to(device=device)

-------------------------------------------------------------------------------------------------------
### Training/Evaluation Loop
def main(
    block_fn, 
    data_collate_fn=collate_with_padding,
    pad_idx=None,
    train=True,
    compile=False
):
    torch.random.manual_seed(0)
    device = torch.device(DEVICE)
    torch.set_float32_matmul_precision("high")

    # Create dataset and dataloader
    data_set = FakeDataset()
    data_loader = DataLoader(
        data_set,
        batch_size=BATCH_SIZE,
        collate_fn=data_collate_fn,
        num_workers=12,
        pin_memory=True,
        drop_last=True
    )

    model = MyDecoder(
        block_fn=block_fn,
        num_tokens=NUM_TOKENS,
        dim=DIM,
        num_heads=NUM_HEADS,
        num_layers=DEPTH,
        max_seq_len=MAX_SEQ_LEN,
        pad_idx=pad_idx
    ).to(device)

    if compile:
        model = torch.compile(model)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.SGD(model.parameters())

    def train_step(model, inputs, targets, 
                   position_ids=None, attn_mask=None):
        with torch.amp.autocast(DEVICE, dtype=torch.bfloat16):
            outputs = model(inputs, position_ids, attn_mask)
            outputs = outputs.view(-1, NUM_TOKENS)
            targets = targets.flatten()
            loss = criterion(outputs, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    @torch.no_grad()
    def eval_step(model, inputs, targets, 
                  position_ids=None, attn_mask=None):
        with torch.amp.autocast(DEVICE, dtype=torch.bfloat16):
            outputs = model(inputs, position_ids, attn_mask)
            if outputs.is_nested:
                outputs = outputs.data._values
                targets = targets.data._values
            else:
                outputs = outputs.view(-1, NUM_TOKENS)
                targets = targets.flatten()
            loss = criterion(outputs, targets)
        return loss

    if train:
        model.train()
        step_fn = train_step
    else:
        model.eval()
        step_fn = eval_step

    t0 = time.perf_counter()
    summ = 0
    count = 0

    for step, data in enumerate(data_loader):
        # Copy data to GPU
        data = data_to_device(data, device=device)
        step_fn(model, data['inputs'], data['targets'],
                       position_ids=data.get('indices'),
                       attn_mask=data.get('attn_mask'))

        # Capture step time
        batch_time = time.perf_counter() - t0
        if step > 20:  # Skip first steps
            summ += batch_time
            count += 1
        t0 = time.perf_counter()
        if step >= 100:
            break
    print(f'average step time: {summ / count}')

-------------------------------------------------------------------------------------------------------
### PyTorch SDPA with Padding
from torch.nn.functional import scaled_dot_product_attention as sdpa
block_fn = functools.partial(MyAttentionBlock, attn_fn=sdpa)
causal_block_fn = functools.partial(
    MyAttentionBlock,
    attn_fn=functools.partial(sdpa, is_causal=True)
)

for mode in ['eval', 'train']:
    for compile in [False, True]:
        block_func = causal_block_fn\
            if mode == 'train' else block_fn
        print(f'{mode} with {collate}, '
              f'{"compiled" if compile else "uncompiled"}')
        main(block_fn=block_func,
             pad_idx=PAD_ID,
             train=mode=='train',
             compile=compile)

-------------------------------------------------------------------------------------------------------
### Optimizing for Variable Length Input
# 1. Padding Optimization
def collate_pad_to_longest(batch):
    padded_inputs = []
    padded_targets = []
    max_length = max([b[0].shape[0] for b in batch])
    for b in batch:
        padded_inputs.append(pad_sequence(b[0], max_length, PAD_ID))
        padded_targets.append(pad_sequence(b[1], max_length, PAD_ID))
    padded_inputs = torch.stack(padded_inputs, dim=0)
    padded_targets = torch.stack(padded_targets, dim=0)
    return {
        'inputs': padded_inputs,
        'targets': padded_targets
    }

for mode in ['eval', 'train']:
    for compile in [False, True]:
        block_func = causal_block_fn\
            if mode == 'train' else block_fn
        print(f'{mode} with {collate}, '
              f'{"compiled" if compile else "uncompiled"}')
        main(block_fn=block_func,
             data_collate_fn=collate_pad_to_longest,
             pad_idx=PAD_ID,
             train=mode=='train',
             compile=compile)

-------------------------------------------------------------------------------------------------------
# 2. SDPA with PyTorch NestedTensors
def nested_tensor_collate(batch):
    inputs = torch.nested.as_nested_tensor([b[0] for b in batch],
                                           layout=torch.jagged)
    targets = torch.nested.as_nested_tensor([b[1] for b in batch],
                                            layout=torch.jagged)
    indices = torch.concat([torch.arange(b[0].shape[0]) for b in batch])

    # workaround for creating a NestedTensor with identical "jagged" shape
    xx = torch.empty_like(inputs)
    xx.data._values[:] = indices

    return {
        'inputs': inputs,
        'targets': targets,
        'indices': xx
    }

for compile in [False, True]:
    print(f'eval with nested tensors, '
          f'{"compiled" if compile else "uncompiled"}')
    main(
        block_fn=block_fn,
        data_collate_fn=nested_tensor_collate,
        train=False,
        compile=compile
    )

-------------------------------------------------------------------------------------------------------
# 3. FlashAttention2
def collate_concat(batch):
    inputs = torch.concat([b[0] for b in batch]).unsqueeze(0)
    targets = torch.concat([b[1] for b in batch]).unsqueeze(0)
    indices = torch.concat([torch.arange(b[0].shape[0]) for b in batch])
    seqlens = torch.tensor([b[0].shape[0] for b in batch])
    seqlens = torch.cumsum(seqlens, dim=0, dtype=torch.int32)
    cu_seqlens = torch.nn.functional.pad(seqlens, (1, 0))

    return {
        'inputs': inputs,
        'targets': targets,
        'indices': indices,
        'attn_mask': cu_seqlens
    }

from flash_attn import flash_attn_varlen_func
fa_varlen = lambda q, k, v, attn_mask: flash_attn_varlen_func(
    q.squeeze(0),
    k.squeeze(0),
    v.squeeze(0),
    cu_seqlens_q=attn_mask,
    cu_seqlens_k=attn_mask,
    max_seqlen_q=MAX_SEQ_LEN,
    max_seqlen_k=MAX_SEQ_LEN
).unsqueeze(0)

fa_varlen_causal = lambda q, k, v, attn_mask: flash_attn_varlen_func(
    q.squeeze(0),
    k.squeeze(0),
    v.squeeze(0),
    cu_seqlens_q=attn_mask,
    cu_seqlens_k=attn_mask,
    max_seqlen_q=MAX_SEQ_LEN,
    max_seqlen_k=MAX_SEQ_LEN,
    causal=True
).unsqueeze(0)

block_fn = functools.partial(MyAttentionBlock,
                             attn_fn=fa_varlen,
                             format='bshd')

causal_block_fn = functools.partial(MyAttentionBlock,
                                    attn_fn=fa_varlen_causal,
                                    format='bshd')

print('flash-attn eval')
main(
    block_fn=block_fn,
    data_collate_fn=collate_concat,
    train=False
)

print('flash-attn train')
main(
    block_fn=causal_block_fn,
    data_collate_fn=collate_concat,
    train=True,
)

-------------------------------------------------------------------------------------------------------
# 4. XFormers Memory Efficient Attention
from xformers.ops import fmha
from xformers.ops import memory_efficient_attention as mea

def collate_xformer(batch):
    inputs = torch.concat([b[0] for b in batch]).unsqueeze(0)
    targets = torch.concat([b[1] for b in batch]).unsqueeze(0)
    indices = torch.concat([torch.arange(b[0].shape[0]) for b in batch])
    seqlens = [b[0].shape[0] for b in batch]
    batch_sizes = [1 for b in batch]
    block_diag = fmha.BlockDiagonalMask.from_seqlens(seqlens, device='cpu')
    block_diag._batch_sizes = batch_sizes

    return {
        'inputs': inputs,
        'targets': targets,
        'indices': indices,
        'attn_mask': block_diag
    }

mea_eval = lambda q, k, v, attn_mask: mea(
    q,k,v, attn_bias=attn_mask)

mea_train = lambda q, k, v, attn_mask: mea(
    q,k,v, attn_bias=attn_mask.make_causal())

block_fn = functools.partial(MyAttentionBlock,
                             attn_fn=mea_eval,
                             format='bshd')

causal_block_fn = functools.partial(MyAttentionBlock,
                             attn_fn=mea_train,
                             format='bshd')

print(f'xFormer Attention ')
for compile in [False, True]:
    print(f'eval with xFormer Attention, '
          f'{"compiled" if compile else "uncompiled"}')
    main(block_fn=block_fn,
         train=False,
         data_collate_fn=collate_xformer,
         compile=compile)

print(f'train with xFormer Attention')
main(block_fn=causal_block_fn,
     train=True,
     data_collate_fn=collate_xformer)

-------------------------------------------------------------------------------------------------------
# GPT2LMHeadModel
from transformers import GPT2Config, GPT2LMHeadModel

# Use random data
class HuggingFaceFakeDataset(Dataset):
    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        length = torch.randint(1, MAX_SEQ_LEN, (1,))
        input_ids = torch.randint(1, NUM_TOKENS, (length,))
        labels = input_ids.clone()
        labels[0] = PAD_ID # ignore first token
        return {
            'input_ids': input_ids,
            'labels': labels
        }
        return input_ids, labels

def hf_collate_with_padding(batch):
    padded_inputs = []
    padded_labels = []
    for b in batch:
        input_ids = b['input_ids']
        labels = b['labels']
        padded_inputs.append(pad_sequence(input_ids, MAX_SEQ_LEN, PAD_ID))
        padded_labels.append(pad_sequence(labels, MAX_SEQ_LEN, PAD_ID))
    padded_inputs = torch.stack(padded_inputs, dim=0)
    padded_labels = torch.stack(padded_labels, dim=0)
    return {
        'input_ids': padded_inputs,
        'labels': padded_labels,
        'attention_mask': (padded_inputs != PAD_ID)
    }

def hf_main(
    config,
    collate_fn=hf_collate_with_padding,
    compile=False
):
    torch.random.manual_seed(0)
    device = torch.device(DEVICE)
    torch.set_float32_matmul_precision("high")

    # Create dataset and dataloader
    data_set = HuggingFaceFakeDataset()
    data_loader = DataLoader(
        data_set,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn,
        num_workers=12 if DEVICE == "CUDA" else 0,
        pin_memory=True,
        drop_last=True
    )

    model = GPT2LMHeadModel(config).to(device)

    if compile:
        model = torch.compile(model)

    # Define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.SGD(model.parameters())

    model.train()

    t0 = time.perf_counter()
    summ = 0
    count = 0

    for step, data in enumerate(data_loader):
        # Copy data to GPU
        data = data_to_device(data, device=device)
        input_ids = data['input_ids']
        labels = data['labels']
        position_ids = data.get('position_ids')
        attn_mask = data.get('attention_mask')
        with torch.amp.autocast(DEVICE, dtype=torch.bfloat16):
            outputs = model(input_ids=input_ids,
                            position_ids=position_ids,
                            attention_mask=attn_mask)
            logits = outputs.logits[..., :-1, :].contiguous()
            labels = labels[..., 1:].contiguous()
            loss = criterion(logits.view(-1, NUM_TOKENS), labels.flatten())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Capture step time
        batch_time = time.perf_counter() - t0
        if step > 20:  # Skip first steps
            summ += batch_time
            count += 1
        t0 = time.perf_counter()
        if step >= 100:
            break
    print(f'average step time: {summ / count}')
  
-------------------------------------------------------------------------------------------------------
# SDPA with Padding
config = GPT2Config(
        n_layer=DEPTH,
        n_embd=DIM,
        n_head=NUM_HEADS,
        vocab_size=NUM_TOKENS,
    )

for compile in [False, True]:
    print(f"HF GPT2 train with SDPA, compile={compile}")
    hf_main(config=config, compile=compile)

-------------------------------------------------------------------------------------------------------
# FlashAttention2
flash_config = GPT2Config(
        n_layer=DEPTH,
        n_embd=DIM,
        n_head=NUM_HEADS,
        vocab_size=NUM_TOKENS,
        attn_implementation='flash_attention_2'
    )

print(f"HF GPT2 train with flash")
hf_main(config=flash_config)

-------------------------------------------------------------------------------------------------------
# FlashAttention2 with Unpadded Input
def collate_flatten(batch):
    input_ids = torch.concat([b['input_ids'] for b in batch]).unsqueeze(0)
    labels = torch.concat([b['labels'] for b in batch]).unsqueeze(0)
    position_ids = [torch.arange(b['input_ids'].shape[0]) for b in batch]
    position_ids = torch.concat(position_ids)

    return {
        'input_ids': input_ids,
        'labels': labels,
        'position_ids': position_ids
    }

print(f"HF GPT2 train with flash, no padding")
hf_main(config=flash_config, collate_fn=collate_flatten)

