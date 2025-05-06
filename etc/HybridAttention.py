### From https://pub.towardsai.net/hybrid-attention-for-binary-sequence-forecasting-637abff3f87d

"""
1. Decoding Binary Sequence Dynamics: Summary
   This work presents a predictive modeling framework designed to forecast both the next bit in a binary time series 
   and the number of 1s in the near future. 
   It combines symbolic pattern recognition, statistical recency analysis, and Transformer-based attention mechanisms
   to capture both short- and long-term sequence dynamics.

2. Input: Binary Time Series and Feature Encoding
   Given a binary time series {𝑥_𝑡}, where each value is either 0 or 1 (e.g., machine faults, user clicks), the task is to predict:
       -1. The probability of the next bit being 1.
       -2. The count of 1s in the next 𝐾 steps.

   -a. Two Complementary Encodings of Past Data:
        -1. n-gram Tokens:
            -1) For each time step, the past 𝑛 bits (𝑥_(𝑡−𝑛),...,𝑥_(𝑡−1)) form a symbolic “word” (n-gram).
            -2) Each n-gram is mapped to an integer index in a vocabulary 𝑉, capturing motifs like “010”.
        -2. Recency Statistics:
            -1) Within sliding windows 𝑤∈𝑊, two features are computed:
                -1. Local mean: the proportion of 1s.
                -2. Flip-rate: frequency of transitions (0→1 or 1→0).
            -2) These describe local momentum and volatility in the sequence.
       Together, these encodings form a tuple (𝑝_𝑡,𝑟_𝑡) — a compact summary of recent binary behavior.

3. Fusion via Self-Attention: The TrendFormer Architecture
   -a. Embedding:
       -1. n-gram indices 𝑝_𝑡 are embedded into dense vectors via matrix 𝐸
   -b. Self-Attention:
       -1. A single attention block processes the stack of n-gram embeddings, yielding attention outputs:
           𝑧ˉ=Avg(Self-Attn(𝐸(𝑝_𝑡)))
       -2. This produces a nonlinear, data-driven summary capturing higher-order dependencies, akin to a learned Markov model.
   -c. Recency MLP:
       -1. The numeric recency vector 𝑟_𝑡 is passed through a small multilayer perceptron to yield a dense vector 𝑟^_𝑡
   -d. Fusion:
       𝑧ˉ and 𝑟^_𝑡 are concatenated and passed through a dense layer with ReLU activation to form a unified hidden state ℎ_𝑡:
       ℎ_𝑡=𝜙(𝑊[𝑧ˉ;𝑟^_𝑡])

4. Dual-Head Output Structure
   -a. Point Head:
       -1. Outputs a scalar logit 𝑠_𝑡, converted to the next-bit probability 𝑝_𝑡=𝜎(𝑠_𝑡)
   -b. Count Head:
       -1. Outputs a vector 𝑢_𝑡, passed through a Softmax to produce a categorical distribution:
           Pr(𝐶_𝑡=𝑐),  for 𝑐=0,...,𝐾
       -2. Predicts how many 1s will occur over the next 𝐾 steps.

5. Training: Multi-Task Loss and Temporal Validation
   -a. Loss Function:
       -1. Composite of:
           -1) Bernoulli loss for the next-bit prediction.
           -2) Categorical loss for the count distribution.
       -2. Shared internal state ℎ_𝑡 allows both tasks to reinforce each other:
           -1) Point prediction keeps the model sharp for local events.
           -2) Count prediction encourages recognition of long-term patterns.
   -b. Validation Strategy:
       -1. Forward-expanding cross-validation: ensures causal integrity by using only past data to predict future data.
       -2. Evaluation metrics:
           -1) Next-bit: Log-loss, Brier score, AUROC.
           -2) Count: Ranked Probability Score.
           -3) Coverage check: compares empirical vs predicted 95% count intervals.

6. Uncertainty Modeling and Benchmark Comparison
   -a. Example: A factory sensor with a 20% fault rate.
       -1. Naïve model: Uses static Bernoulli assumptions → fails to update after streaks, lacks distributional richness.
       -2. This model:
           -1) Responds to rare n-grams (e.g., "111" signaling potential failure).
           -2) Adjusts forecasts dynamically with recency statistics.
           -3) Provides probabilistic ranges — e.g., 32% chance of 3+ faults with a 0–4 count interval, vs 6% and 0–3 in a binomial model.
       -3. Result: Provides early warnings and better-calibrated uncertainty.

7. Full Pipeline Summary
   -a. Encode past as symbolic (n-grams) + statistical (recency) features.
   -b. Fuse them via attention + MLP to capture both motifs and momentum.
   -c. Predict:
       -1. The next bit (point prediction).
       -2. The count of 1s in the next 𝐾 steps (distributional prediction).
   -d. Optimize with joint loss under time-series cross-validation.
   -e. Validate using accuracy, probabilistic scoring, and interval coverage checks.
"""

import warnings
warnings.filterwarnings("ignore")

import math
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

def ranked_probability_score(true_counts, cdf_preds):
    K = cdf_preds.shape[1] - 1
    return np.mean(np.sum((cdf_preds - (np.arange(K+1)[None, :] >= true_counts[:, None]))**2, axis=1))

class BinarySeriesDataset(Dataset):
    def __init__(self, series, K, ngram_vocab, recency_windows):
        self.series = series
        self.K = K
        self.vocab = ngram_vocab
        self.wins = recency_windows
        self.ngram_sizes = sorted({len(p) for p in ngram_vocab})
        self.offset = max(self.K, max(self.ngram_sizes), max(self.wins))
        self.length = len(series) - self.offset

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        t = idx + self.offset
        future = self.series[t:t+self.K]
        y_point = float(self.series[t])
        y_count = int(future.sum())
        patterns = []
        for n in self.ngram_sizes:
            pat = tuple(int(x) for x in self.series[t-n:t])
            patterns.append(self.vocab.get(pat, 0))
        patterns = np.array(patterns, dtype=np.int64)
        rec_feats = []
        for w in self.wins:
            window = self.series[t-w:t]
            rec_feats.append(window.mean())
            rec_feats.append((window[1:] != window[:-1]).mean() if w>1 else 0.0)
        rec_feats = np.array(rec_feats, dtype=np.float32)
        return patterns, rec_feats, y_point, y_count

class BinaryTrendFormer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, recency_dim, hidden_dim, K, dropout_p=0.2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.rec_fc = nn.Sequential(nn.Linear(recency_dim, hidden_dim), nn.ReLU())
        self.fuse_fc = nn.Sequential(nn.Linear(embed_dim+hidden_dim, hidden_dim), nn.ReLU())
        self.dropout = nn.Dropout(dropout_p)
        self.point_head = nn.Linear(hidden_dim, 1)
        self.count_head = nn.Linear(hidden_dim, K+1)

    def forward(self, patterns, rec_feats):
        x = self.embed(patterns)
        attn_out, _ = self.attn(x, x, x)
        pat_feat = attn_out.mean(dim=1)
        rec_feat = self.rec_fc(rec_feats)
        fusion = torch.cat([pat_feat, rec_feat], dim=1)
        h = self.fuse_fc(fusion)
        h = self.dropout(h)
        p_logit = self.point_head(h).squeeze(-1)
        p_point = torch.sigmoid(p_logit)
        count_logits = self.count_head(h)
        return p_point, p_logit, count_logits

def train_and_evaluate(series, K=5,
                       ngram_range=(3,7),
                       recency_windows=[3,7,14],
                       embed_dim=64,
                       num_heads=4,
                       hidden_dim=128,
                       lr=1e-3,
                       weight_decay=1e-5,
                       batch_size=128,
                       epochs=20,
                       patience=5,
                       tscv_splits=3,
                       device='cpu'):

    # Prepare vocabulary
    min_n, max_n = ngram_range
    patterns = [tuple(series[t-n:t]) for t in range(max_n, len(series)) for n in range(min_n, max_n+1)]
    vocab = {pat: i+1 for i, pat in enumerate(set(patterns))}
    recency_dim = 2 * len(recency_windows)
    vocab_size = len(vocab) + 1

    tscv = TimeSeriesSplit(n_splits=tscv_splits)
    metrics = {'logloss':[], 'brier':[], 'auroc':[], 'rps':[], 'coverage':[]}
    last_model = None

    for train_idx, val_idx in tscv.split(series):
        tr, vl = series[train_idx], series[val_idx]
        tr_ds = BinarySeriesDataset(tr, K, vocab, recency_windows)
        vl_ds = BinarySeriesDataset(vl, K, vocab, recency_windows)
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size)

        model = BinaryTrendFormer(vocab_size, embed_dim, num_heads,
                                  recency_dim, hidden_dim, K).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=3, factor=0.5)
        bce = nn.BCEWithLogitsLoss()
        ce  = nn.CrossEntropyLoss()

        # Early-stop training
        best_val_loss = float('inf')
        wait = 0
        best_state = None

        for epoch in range(epochs):
            model.train()
            for p_np, r_np, y_pt, y_ct in tr_loader:
                p = torch.tensor(p_np, dtype=torch.long, device=device)
                r = torch.tensor(r_np, dtype=torch.float32, device=device)
                yp = torch.tensor(y_pt, dtype=torch.float32, device=device)
                yc = torch.tensor(y_ct, dtype=torch.long, device=device)
                opt.zero_grad()
                _, logit, cnt_logits = model(p, r)
                loss = bce(logit, yp) + ce(cnt_logits, yc)
                loss.backward()
                opt.step()

            # Validation loss (batch-wise)
            model.eval()
            val_losses = []
            with torch.no_grad():
                for p_np, r_np, y_pt, _ in vl_loader:
                    p = torch.tensor(p_np, dtype=torch.long, device=device)
                    r = torch.tensor(r_np, dtype=torch.float32, device=device)
                    yp = torch.tensor(y_pt, dtype=torch.float32, device=device)
                    _, logit, _ = model(p, r)
                    probs = torch.sigmoid(logit).cpu().numpy()
                    val_losses.append(log_loss(yp.cpu().numpy(), probs))
            val_loss = np.mean(val_losses)
            scheduler.step(val_loss)

            if val_loss < best_val_loss - 1e-4:
                best_val_loss = val_loss
                wait = 0
                best_state = model.state_dict()
            else:
                wait += 1
                if wait >= patience:
                    break

        model.load_state_dict(best_state)
        last_model = model

        # Compute fold metrics
        model.eval()
        all_p, all_cdf, all_counts = [], [], []
        with torch.no_grad():
            for p_np, r_np, _, y_ct in vl_loader:
                p = torch.tensor(p_np, dtype=torch.long, device=device)
                r = torch.tensor(r_np, dtype=torch.float32, device=device)
                p_pt, _, cnt_logits = model(p, r)
                all_p.extend(p_pt.cpu().numpy())
                probs = torch.softmax(cnt_logits, dim=1).cpu().numpy()
                all_cdf.extend(np.cumsum(probs, axis=1))
                all_counts.extend(y_ct)

        all_p      = np.array(all_p)
        all_cdf    = np.array(all_cdf)
        all_counts = np.array(all_counts)

        ll  = log_loss((all_counts>0).astype(int), all_p)
        bri = brier_score_loss((all_counts>0).astype(int), all_p)
        auc = roc_auc_score((all_counts>0).astype(int), all_p)
        rps = ranked_probability_score(all_counts, all_cdf)

        mean_cdf = all_cdf.mean(axis=0)
        low_idx = np.searchsorted(mean_cdf, 0.025, side='left')
        high_idx = np.searchsorted(mean_cdf, 0.975, side='left')
        cdf_low = all_cdf[:, low_idx]
        cdf_high = all_cdf[:, high_idx]
        cov = np.mean((all_counts>=cdf_low) & (all_counts<=cdf_high))

        metrics['logloss'].append(ll)
        metrics['brier'].append(bri)
        metrics['auroc'].append(auc)
        metrics['rps'].append(rps)
        metrics['coverage'].append(cov)

    # Validation summary
    for k in metrics:
        metrics[k] = np.mean(metrics[k])
    print("\n=== Validation Summary ===")
    print(f"1-step → LogLoss {metrics['logloss']:.3f}, Brier {metrics['brier']:.3f}, AUC {metrics['auroc']:.3f}")
    print(f"K-step → RPS {metrics['rps']:.3f}, 95% CI Cov {metrics['coverage']:.3f}\n")

    # Retrain on full series
    final_model = BinaryTrendFormer(vocab_size, embed_dim, num_heads,
                                    recency_dim, hidden_dim, K).to(device)
    opt = optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    bce = nn.BCEWithLogitsLoss()
    ce  = nn.CrossEntropyLoss()
    full_loader = DataLoader(BinarySeriesDataset(series, K, vocab, recency_windows),
                              batch_size=batch_size, shuffle=True)
    final_model.train()
    for _ in range(epochs):
        for p_np, r_np, y_pt, y_ct in full_loader:
            p = torch.tensor(p_np, dtype=torch.long, device=device)
            r = torch.tensor(r_np, dtype=torch.float32, device=device)
            yp = torch.tensor(y_pt, dtype=torch.float32, device=device)
            yc = torch.tensor(y_ct, dtype=torch.long, device=device)
            opt.zero_grad()
            _, logit, cnt_logits = final_model(p, r)
            (bce(logit, yp) + ce(cnt_logits, yc)).backward()
            opt.step()

    # Final forecast and intervals
    final_model.eval()
    with torch.no_grad():
        ds = BinarySeriesDataset(series, K, vocab, recency_windows)
        p_np, r_np, _, _ = ds[-1]
        p = torch.tensor(p_np, dtype=torch.long, device=device).unsqueeze(0)
        r = torch.tensor(r_np, dtype=torch.float32, device=device).unsqueeze(0)
        p_pt, _, cnt_logits = final_model(p, r)
        p_next = float(p_pt.cpu())
        dist = torch.softmax(cnt_logits, dim=1).cpu().numpy().flatten()
        exp_count = np.dot(np.arange(K+1), dist)

        cdf = np.cumsum(dist)
        idx_low = np.searchsorted(cdf, 0.025, side='left')
        idx_high = np.searchsorted(cdf, 0.975, side='left')
        idx_low = min(max(idx_low, 0), K)
        idx_high = min(max(idx_high, 0), K)
        model_cl_low = cdf[idx_low]
        model_cl_hi  = cdf[idx_high]

        # Bernoulli-CLT intervals
        p_hat = series.mean()
        n_hat = len(series)
        se_p = math.sqrt(p_hat*(1-p_hat)/n_hat)
        clt_low = max(0, p_hat - 1.96*se_p)
        clt_hi  = min(1, p_hat + 1.96*se_p)
        mu_c = K * p_hat
        se_c = math.sqrt(K * p_hat * (1-p_hat))
        count_low = max(0, mu_c - 1.96*se_c)
        count_hi  = min(K, mu_c + 1.96*se_c)

    print(f"Final Next-step P(1): {p_next:.3f}")
    print(f" Model CDF 95% CI for P(1): [{model_cl_low:.3f}, {model_cl_hi:.3f}]")
    print(f" Bernoulli-CLT 95% CI for p: [{clt_low:.3f}, {clt_hi:.3f}] (n={n_hat})")
    print(f"Final Expected #1s in next {K}: {exp_count:.2f}")
    print(f" Model CDF 95% CI for count: [{model_cl_low*K:.2f}, {model_cl_hi*K:.2f}]")
    print(f" Bernoulli-CLT 95% CI for count: [{count_low:.2f}, {count_hi:.2f}]\n")

if __name__ == "__main__":
    df_loaded = pd.read_csv('bin_series.csv')
    series = df_loaded['value'].values
    train_and_evaluate(series, K=5, epochs=20, tscv_splits=3, device='cpu')
