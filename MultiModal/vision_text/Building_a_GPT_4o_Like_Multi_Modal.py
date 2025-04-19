### From https://levelup.gitconnected.com/building-a-gpt-4o-like-multi-modal-from-scratch-using-python-ad0fa9c213d3
### Have to see given link

# =========================
# 0. Imports & Utilities
# =========================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

import re
import collections
import math
import os
import numpy as np

# -------------------------
# Regex for word / punctuation split
# -------------------------
SPLIT_PATTERN = r'\w+|[^\s\w]+'


# =========================
# 1.  BPE Tokenizer Trainer
# =========================
def learn_bpe(corpus: str, num_merges: int = 75, eos_token: str = "</w>"):
    corpus = corpus.lower()
    words = re.findall(SPLIT_PATTERN, corpus)
    freqs = collections.Counter(words)

    # word -> list(chars + </w>)
    rep = {w: list(w) + [eos_token] for w in freqs}
    vocab = {c for chars in rep.values() for c in chars}

    merges = {}
    for i in range(num_merges):
        pair_counts = collections.Counter()
        for w, f in freqs.items():
            chars = rep[w]
            for j in range(len(chars) - 1):
                pair_counts[(chars[j], chars[j + 1])] += f
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        merges[best] = i
        new_sym = "".join(best)

        for w in rep:
            chars = rep[w]
            j = 0
            merged = []
            while j < len(chars):
                if j < len(chars) - 1 and (chars[j], chars[j + 1]) == best:
                    merged.append(new_sym)
                    j += 2
                else:
                    merged.append(chars[j])
                    j += 1
            rep[w] = merged
        vocab.add(new_sym)

    return merges, vocab, rep


def bpe_tokenize(text: str, merges: dict, eos_token: str = "</w>"):
    tokens = re.findall(SPLIT_PATTERN, text.lower())
    out = []
    for tok in tokens:
        chars = list(tok) + [eos_token]
        i = 0
        while i < len(chars) - 1:
            pair = (chars[i], chars[i + 1])
            if pair in merges:
                chars[i : i + 2] = ["".join(pair)]
            else:
                i += 1
        out.extend(chars)
    return out


# =========================
# 2.  Character‑Level GPT
# =========================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, d_ff)
        self.ff2 = nn.Linear(d_ff, d_model)
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

    def forward(self, x, mask):
        B, T, C = x.shape
        qkv = (
            self.qkv(self.ln1(x))
            .view(B, T, self.n_heads, 3 * self.d_k)
            .permute(0, 2, 1, 3)
        )
        q, k, v = qkv.chunk(3, dim=-1)
        scores = (q @ k.transpose(-2, -1)) * self.d_k ** -0.5
        scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, -1)
        out = (
            (attn @ v)
            .permute(0, 2, 1, 3)
            .contiguous()
            .view(B, T, C)
        )
        x = x + self.proj(out)
        x = x + self.ff2(torch.relu(self.ff1(self.ln2(x))))
        return x


class DecoderOnlyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, block_size):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        pe = torch.zeros(block_size, d_model)
        pos = torch.arange(0, block_size, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_enc[:, :T]
        m = self.mask[:, :, :T, :T]
        for blk in self.blocks:
            x = blk(x, m)
        x = self.ln_f(x)
        return self.head(x)

    @torch.no_grad()
    def generate(self, idx, steps):
        for _ in range(steps):
            logits = self(idx)[:, -1]
            next_id = torch.multinomial(torch.softmax(logits, -1), 1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# =========================
# 3.  Minimal Training Loop
# =========================
def train_gpt(
    model,
    data_tensor,
    block_size,
    epochs=5000,
    batch_size=16,
    lr=3e-4,
    device="cpu",
):
    def get_batches(data):
        xs, ys = [], []
        for i in range(len(data) - block_size):
            xs.append(data[i : i + block_size])
            ys.append(data[i + 1 : i + block_size + 1])
        return torch.stack(xs), torch.stack(ys)

    x, y = get_batches(data_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epochs):
        idx = torch.randint(0, x.size(0), (batch_size,))
        xb, yb = x[idx].to(device), y[idx].to(device)
        loss = criterion(model(xb).view(-1, model.head.out_features), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ep % 500 == 0:
            print(f"epoch {ep:5d} | loss {loss.item():.4f}")


# =========================
# 4.  Vision Feature Extractor
# =========================
class VisionEncoder(nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        base = torchvision.models.resnet18(weights="DEFAULT")
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # remove fc
        self.proj = nn.Linear(base.fc.in_features, target_dim)

    def forward(self, x):
        feats = self.backbone(x).squeeze()
        return self.proj(feats)


# =========================
# 5.  Multi‑Modal Utilities
# =========================
def get_image_feature(img_path, encoder, transform, device="cpu"):
    img = Image.open(img_path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        return encoder(t)


# (Further multimodal training loops for images / video / audio follow the same
# pattern: prepend a special token, replace its embedding with a projected
# feature vector, and fine‑tune with an appropriate loss.)

# ==========================================================
# 6.  BPE  – Full Training / Inference Helpers
# ==========================================================
def bpe_train(corpus: str, merges: int = 75) -> dict:
    merges_dict, vocab, word_repr = learn_bpe(corpus, merges)
    return {
        "merges": merges_dict,
        "vocab": vocab,
        "word_repr": word_repr,
        "eos": "</w>",
    }


def bpe_encode(text: str, bpe_state: dict) -> list[str]:
    return bpe_tokenize(text, bpe_state["merges"], bpe_state["eos"])


def bpe_decode(tokens: list[str]) -> str:
    out = " ".join(t.replace("</w>", "").strip() for t in tokens)
    return out.replace("  ", " ").strip()


# ==========================================================
# 7.  GPT  – Training & Generation Utility
# ==========================================================
def gpt_generate(
    model: DecoderOnlyGPT,
    tokenizer: dict,
    prompt: str,
    steps: int = 200,
    device="cpu",
):
    ids = torch.tensor([[tokenizer[ch] for ch in prompt]], device=device)
    out = model.generate(ids, steps)[0].tolist()
    return "".join([int_to_char[i] for i in out])


# ==========================================================
# 8.  Multi‑Modal  – Dataset Helpers
# ==========================================================
def build_mm_batch(
    batch_indices,
    input_ids,
    attn_masks,
    img_paths,
    img_feats_cache,
    proj_layer,
    num_img_tok,
    pad_id,
    device="cpu",
):
    xb = input_ids[batch_indices].to(device)
    mask = attn_masks[batch_indices].to(device)

    # replace IMG token embeddings
    img_feats = torch.stack(
        [img_feats_cache[p] for p in img_paths[batch_indices]]
    ).to(device)
    proj_feats = proj_layer(img_feats).unsqueeze(1)  # (B,1,d_model)
    return xb, mask, proj_feats


# ==========================================================
# 9.  Multi‑Modal  – Training Loop (Images)
# ==========================================================
def train_multimodal_img(
    model_blocks,
    embedding,
    proj_layer,
    ln_final,
    lm_head,
    positional_encoding,
    causal_mask,
    train_ids,
    train_masks,
    img_paths,
    img_feat_cache,
    pad_id,
    epochs=2000,
    lr=3e-4,
    batch_size=4,
    eval_int=500,
    device="cpu",
):
    params = (
        list(embedding.parameters())
        + list(proj_layer.parameters())
        + [p for blk in model_blocks for p in blk.parameters()]
        + list(ln_final.parameters())
        + list(lm_head.parameters())
    )
    opt = optim.AdamW(params, lr=lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for ep in range(epochs):
        idx = torch.randint(0, train_ids.size(0), (batch_size,))
        xb_ids, mask = train_ids[idx], train_masks[idx]
        xb_ids, mask = xb_ids.to(device), mask.to(device)
        xb_imgs = [img_paths[i] for i in idx.tolist()]

        # prepare embeddings
        tok_emb = embedding(xb_ids)
        tok_emb[:, :1] = proj_layer(
            torch.stack([img_feat_cache[p] for p in xb_imgs]).to(device)
        ).unsqueeze(1)
        x = tok_emb + positional_encoding[:, : xb_ids.size(1)]
        m = causal_mask[:, :, : x.size(1), : x.size(1)] * mask.unsqueeze(1).unsqueeze(2)

        for blk in model_blocks:
            x = blk(x, m)

        logits = lm_head(ln_final(x))
        tgt = xb_ids.roll(-1, dims=1)
        tgt[:, -1] = -100  # ignore last
        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % eval_int == 0 or ep == epochs - 1:
            print(f"mm‑img | epoch {ep:4d} | loss {loss.item():.4f}")


# ==========================================================
# 10.  Text‑to‑Image  – Training Loop
# ==========================================================
def train_t2i(
    model_blocks,
    embedding,
    ln_final,
    t2i_head,
    positional_encoding,
    causal_mask,
    prompts_ids,
    prompts_masks,
    target_feats,
    epochs=5000,
    lr=1e-4,
    batch_size=4,
    eval_int=500,
    device="cpu",
):
    params = (
        list(embedding.parameters())
        + [p for blk in model_blocks for p in blk.parameters()]
        + list(ln_final.parameters())
        + list(t2i_head.parameters())
    )
    opt = optim.AdamW(params, lr=lr)
    mse = nn.MSELoss()

    for ep in range(epochs):
        idx = torch.randint(0, prompts_ids.size(0), (batch_size,))
        xb, mask = prompts_ids[idx].to(device), prompts_masks[idx].to(device)
        yb = target_feats[idx].to(device)

        x = embedding(xb) + positional_encoding[:, : xb.size(1)]
        m = causal_mask[:, :, : xb.size(1), : xb.size(1)] * mask.unsqueeze(1).unsqueeze(
            2
        )
        for blk in model_blocks:
            x = blk(x, m)

        last_idx = mask.sum(1) - 1
        hid = x[torch.arange(x.size(0)), last_idx]  # (B,d_model)
        pred = t2i_head(hid)
        loss = mse(pred, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ep % eval_int == 0 or ep == epochs - 1:
            print(f"t2i | epoch {ep:4d} | mse {loss.item():.6f}")


# ==========================================================
# 11.  Nearest‑Neighbor  – Feature to Image
# ==========================================================
def nearest_img(pred_feat: torch.Tensor, db: list[tuple[str, torch.Tensor]]):
    best, best_d = None, 1e9
    for path, feat in db:
        d = torch.dist(pred_feat, feat)
        if d < best_d:
            best, best_d = path, d
    return best, best_d

# ==========================================================
# 12.  Positional Encoding / Causal Mask Helpers
# ==========================================================
def build_positional_encoding(block_size: int, d_model: int, device="cpu"):
    pe = torch.zeros(block_size, d_model, device=device)
    pos = torch.arange(0, block_size, dtype=torch.float, device=device).unsqueeze(1)
    div = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # (1, T, C)


def build_causal_mask(block_size: int, device="cpu"):
    return torch.tril(torch.ones(block_size, block_size, device=device)).view(
        1, 1, block_size, block_size
    )


# ==========================================================
# 13.  Vision Feature Cache
# ==========================================================
def cache_image_features(img_paths, encoder, transform, device="cpu"):
    cache = {}
    for path in img_paths:
        if path not in cache:
            cache[path] = get_image_feature(path, encoder, transform, device).squeeze()
    return cache


# ==========================================================
# 14.  Multi‑Modal Generation (Images)
# ==========================================================
@torch.no_grad()
def mm_generate_image_response(
    prompt,
    img_path,
    tokenizer,
    embedding,
    proj_layer,
    transformer_blocks,
    ln_final,
    lm_head,
    positional_encoding,
    causal_mask,
    block_size,
    device="cpu",
    max_tokens=100,
):
    # ---- prepare inputs ----
    img_tok_id = tokenizer["<IMG>"]
    pad_id = tokenizer["<PAD>"]
    ids = [img_tok_id] + [tokenizer.get(ch, pad_id) for ch in prompt]
    if len(ids) > block_size:
        ids = ids[:block_size]
    pad_len = block_size - len(ids)
    attn_mask = [1] * len(ids) + [0] * pad_len
    ids = ids + [pad_id] * pad_len

    idx = torch.tensor([ids], device=device)
    mask = torch.tensor([attn_mask], device=device)

    # ---- embed + replace image token ----
    tok_emb = embedding(idx)
    img_feat = get_image_feature(img_path, vision_encoder, transforms_pipeline, device)
    tok_emb[:, 0] = proj_layer(img_feat)

    x = tok_emb + positional_encoding[:, : block_size]
    ca_mask = causal_mask[:, :, : block_size, : block_size] * mask.unsqueeze(1).unsqueeze(2)

    # ---- pass through transformer ----
    for blk in transformer_blocks:
        x = blk(x, ca_mask)
    logits = lm_head(ln_final(x))  # (1, T, V)

    generated = []
    for _ in range(max_tokens):
        last_logit = logits[:, len(prompt), :]  # focus on next token position
        next_id = torch.multinomial(torch.softmax(last_logit, -1), 1).item()
        if next_id == pad_id:
            break
        generated.append(tokenizer["inv"][next_id])
    return "".join(generated)


# ==========================================================
# 15.  Text‑to‑Image Generation + Retrieval
# ==========================================================
@torch.no_grad()
def text_to_image_retrieval(
    prompt_text: str,
    tokenizer: dict,
    embedding,
    transformer_blocks,
    ln_final,
    t2i_head,
    positional_encoding,
    causal_mask,
    block_size,
    known_feature_db: list[tuple[str, torch.Tensor]],
    device="cpu",
):
    # tokenize / pad
    pad_id = tokenizer["<PAD>"]
    ids = [tokenizer.get(ch, pad_id) for ch in prompt_text]
    ids = ids[: block_size] + [pad_id] * (block_size - len(ids))
    mask = [1 if i < len(prompt_text) else 0 for i in range(block_size)]

    idx = torch.tensor([ids], device=device)
    msk = torch.tensor([mask], device=device)

    x = embedding(idx) + positional_encoding[:, : block_size]
    ca_mask = causal_mask[:, :, : block_size, : block_size] * msk.unsqueeze(1).unsqueeze(2)
    for blk in transformer_blocks:
        x = blk(x, ca_mask)

    last_hidden = x[0, msk.sum() - 1]  # (d_model,)
    pred_feat = t2i_head(ln_final(last_hidden.unsqueeze(0))).squeeze()

    # nearest neighbor retrieval
    best_path, best_dist = nearest_img(pred_feat, known_feature_db)
    return best_path, best_dist.item()


# ==========================================================
# 16.  Audio / Video Stub Extractors
# ==========================================================
class DummyAudioExtractor(nn.Module):
    """maps MFCC (num_coeffs) -> 256‐dim feature"""
    def __init__(self, num_coeffs=13, out_dim=256):
        super().__init__()
        self.fc = nn.Linear(num_coeffs, out_dim)

    def forward(self, mfccs):
        return self.fc(mfccs)  # (time, out_dim)


def average_time_features(feat_seq: torch.Tensor):
    """feat_seq: (T, D) → (D,) averaged"""
    return feat_seq.mean(0)


# ==========================================================
# 17.  Conceptual Video Frame Sampler
# ==========================================================
def sample_video_frames(video_reader, num_frames=16):
    """
    Given a torchvision.VideoReader (or similar), 
    uniformly sample `num_frames` PIL Images.
    """
    total = len(video_reader)
    idxs = np.linspace(0, total - 1, num_frames, dtype=int)
    return [video_reader[i]["data"] for i in idxs]  # list of tensors or PIL


