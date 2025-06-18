### From https://pub.towardsai.net/temporal-graph-neural-networks-for-multi-product-time-series-forecasting-f4cc87f8354c

"""
1. Introduction & Motivation
   -a. Scenario: Forecast daily sales for dozens of SKUs in a supply chain where products interact (a spike in one ripples to others).
   -b. Limitation of classical methods (ARIMA, SES):
       -1. Treat each SKU in isolation
       -2. Assume linear trends and stationarity
       -3. Hard to incorporate exogenous signals (promos, holidays, weather)
   -c. Temporal GNN solution:
       -1. Learn a sparse influence graph 𝐴 from data (who influences whom).
       -2. Graph convolutions blend neighbor information per day.
       -3. Temporal convolutions capture evolving patterns across days.
   -d. Applications: electricity demand forecasting, traffic flow, e-commerce recommendations.
   -e. Innovations discussed: sparse‐graph regularization, uncertainty estimates, adaptive graph updates, real‐time retraining.

2. GNN & Temporal GNN Fundamentals
   -a. Graph 𝐺=(𝑉,𝐸): nodes 𝑣_𝑖= SKUs; edges (𝑖,𝑗) with weight 𝑎_𝑖𝑗  = learned influence.
   -b. Node features 𝑥_𝑖: e.g. yesterday’s sales, stock level, calendar encodings.
   -c. Hidden state ℎ_𝑖: blends 𝑥_𝑖 with neighbor messages.

   2.1 Message Passing Neural Networks (MPNNs)
       At layer 𝑘, each node 𝑖 does:
       -a. Message from each neighbor 𝑗:
           𝑚^(𝑘)_(𝑖𝑗)=𝑓(ℎ_𝑖^(𝑘−1), ℎ_𝑗^(𝑘−1),𝑎_𝑖𝑗)
       -b. Aggregate incoming messages:
           𝑀^(𝑘)_𝑖=AGG({𝑚^(𝑘)_𝑖𝑗 : 𝑗∈𝑁(𝑖)})
           where AGG can be sum, mean, max, or attention.
       -c. Update node embedding:
           ℎ^(𝑘)_𝑖=MLP([ℎ^(𝑘-1)_𝑖∥𝑀^(𝑘)_𝑖])
           After 𝐾 layers, optionally a readout ∑_𝑖 ℎ^(𝐾)_𝑖 yields a graph‐level vector.

3. From MLPs to GCNs & GATs
   3.1 Comparing MPNNs vs. MLPs
       -a. MLP: fully connected layers, fixed topology, separate weights per layer.
       -b. MPNN: topology given by learned adjacency 𝐴, shared weight functions 𝑓, scalable to variable node counts.

   3.2 Graph Convolutional Networks (GCNs)
       Extend convolutions to graphs via:
       𝐻^(𝑘)=𝜎(𝐷~^(−1/2) 𝐴~𝐷~^(−1/2) 𝐻^(𝑘−1)𝑊^(𝑘))
       where
       -a. 𝐴~=𝐴+𝐼 adds self-loops
       -b. 𝐷~_𝑖𝑖=∑_𝑗 𝐴~_𝑖𝑗
       -c. 𝑊^(𝑘) is a shared weight matrix,
       -d. 𝜎 = ReLU.
       Properties: weight sharing across edges, captures 1-hop structure per layer, permutation equivariant, and stable via normalized adjacency.

   3.3 Graph Attention Networks (GATs)
       Introduce attention on edges:
       𝛼_𝑢𝑣=softmax_𝑣(𝑒_𝑢𝑣), 𝑒_𝑢𝑣=LeakyReLU(𝑎⊤[𝑊ℎ_𝑢∥𝑊ℎ_𝑣])
       Then
       ℎ′_𝑢=𝜎(∑_(𝑣∈𝑁(𝑢)) 𝛼_𝑢𝑣 𝑊 ℎ_𝑣)
       Benefits: learns which neighbors matter, multi-head variants improve stability and capture multiple relation types, 
                 and attention scores are interpretable.

4. Importance of Sparsity
   -a. Real‐world graphs are sparse—few strong ties matter.
   -b. Techniques:
       -1. L₁ regularization on adjacency logits 𝑍
       -2. Top-k pruning
       -3. Edge sampling
   -c. Loss term:
       𝐿_sparsity = 𝜆 ∥𝜎(𝑍)∥1,
       pushes weak edges 𝑎_𝑖𝑗=𝜎(𝑍_𝑖𝑗) to zero, making the learned graph interpretable and efficient.

5. End-to-End GNN Training
   -a. Inputs:𝑋∈𝑅^(𝑁×𝐹) = node features for 𝑁 SKUs.
   -b. Graph learning: logits 𝑍∈𝑅^(𝑁×𝑁), edges 𝐴=𝜎(𝑍)
   -c. GCN/GAT layers: produce hidden states 𝐻^(𝑘)
   -d. Loss: supervised target loss (e.g. MAE) + sparsity penalty.
   -e. Optimizer: AdamW, learning rate 3×10^−4
   -f. Early stopping on validation MAE.

6. Temporal GNN for Supply-Chain Forecasting
   Combine spatial (graph) and temporal (sequence) modeling:

   6.1 High-Level Flow

   Time t-2      Time t-1      Time t
    ┌───────┐    ┌───────┐     ┌───────┐
    │GNN    │    │GNN    │     │GNN    │   <-- same learned A_t
    └───┬───┘    └───┬───┘     └───┬───┘
        │             │            │
    nodes x_i,t-2 →   x_i,t-1 →   x_i,t
         ...           ...           ...
          ──────────────────────────▶
          Temporal Conv1D (dilated)
          ──────────────────────────▶
          Linear head → next-7 days sales
          
   6.2 Core Components & Code
       ########################################################################################
       class GraphLearner(nn.Module):
           def forward(self, X_t):          # X_t: [N, F_t]
               Z = ...                     # unconstrained logits [N,N]
               A = torch.sigmoid(Z)        # soft adjacency
               return A

       class SpatialEncoder(nn.Module):
           def forward(self, X_t, A):
               H = GCNConv(X_t, A)         # one GCN/GAT layer + ReLU
               return H                    # [N, H]

       class TemporalEncoder(nn.Module):
           def forward(self, H_seq):       # H_seq: [B, N, H, T]
               # reshape → [B, H, N*T] or similar
               Z = Conv1d(..., kernel=3, dilation=2, padding=2)(H_seq)
               return Z                    # [B, H]

       class TemporalGNN(nn.Module):
           def __init__(self, in_dim, hid_dim, horizon):
               super().__init__()
               self.glearner = GraphLearner()
               self.spatial  = SpatialEncoder()
               self.temporal = TemporalEncoder()
               self.head     = nn.Linear(hid_dim, horizon)

           def forward(self, X):           # X: [B, N, F, T]
               H_stack, l1_reg = [], 0.0
               for t in range(X.size(-1)):
                   X_t = X[..., t]         # node features at time t
                   A_t = self.glearner(X_t)
                   H_t = self.spatial(X_t, A_t)
                   H_stack.append(H_t)
                   l1_reg += A_t.abs().mean()  # accumulate sparsity loss

               H_seq = torch.stack(H_stack, dim=-1)  # [B, N, H, T]
               Z     = self.temporal(H_seq)          # [B, H]
               yhat  = self.head(Z)                  # [B, horizon]
               return yhat, l1_reg
       ########################################################################################
       -a. Spatial GCNConv: mixes features via 𝐴_𝑡
       -b. Temporal Conv1d: kernel size = 3, dilation = 2 to capture lags [0,2,4].
       -c. Head: linear mapping 𝑅^𝐻→𝑅^7 for a one‐week horizon.

   6.3 Training Loop (Pseudo)
       for epoch in range(max_epochs):
           for batch in dataloader:
               y_pred, l1 = model(batch.X)
               loss = MAE(y_pred, batch.y) + λ * l1
               loss.backward()
               optimizer.step()
               optimizer.zero_grad()
           validate_and_early_stop()
           
7. Preprocessing & Feature Engineering
   -a. Calendar encoding:
       sin(2𝜋 DOY/365), cos(2𝜋 DOY/365),weekday
   -b. Per-SKU z-score:
       𝑥^=(𝑥−𝜇SKU)/(𝜎SKU)
       avoids a few high-volume SKUs dominating loss.
   -c. Sliding windows:
       Inputs ∈𝑅^(𝐵×𝑁×𝐹×𝐿) with 𝐿=30 days; targets ∈𝑅^(𝐵×7) (only sales).

8. Loss & Metrics
   -a. Objective:
        𝐿=MAE(𝑦^,𝑦)      +     𝜆∥𝜎(𝑍)∥1
             ⏟                     ⏟
       forecast accuracy     sparsity penalty
   -b. Inverse scaling at inference:
       𝑥^=𝑥^_(norm)⋅𝜎+𝜇
   -c. MAPE for portfolio KPI:
       MAPE=100%/7 (∑(𝑡=1 to 𝑡=7)∣(𝑦_𝑡−𝑦^_𝑡)/𝑦_𝑡∣

9. Avoiding Data Leakage
   -a. Chronological split:
       -1. 70% train, 15% val, 15% test.
       -2. Early stopping on validation MAE.

10. Baseline Comparisons
    -a. SARIMAX(1,0,1) per-SKU: can’t directly model cross-SKU effects; at best proxies via autoregressive lags.
    -b. Simple Exponential Smoothing (SES): handles stationary demands but misses coordinated promotional spikes.
    -c. TGNN advantage: learned adjacency 𝐴_𝑖𝑗 instantly propagates SKU-7’s promo spike into SKU-10’s node features, 
                        and the TCN picks up the pattern without manual feature engineering.
"""

import pandas as pd, numpy as np, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler

# 1) load ───────────────────────────────

output_csv_path = "aggregated_sales.csv"
df = pd.read_csv(output_csv_path, parse_dates=["date"]).set_index("date")
# shape: (T, N)  e.g. (800, 50)

# 2) add calendar covariates  (sin/cos, weekday)
cal = pd.DataFrame({
    "sin_doy":  np.sin(2*np.pi*df.index.dayofyear/365),
    "cos_doy":  np.cos(2*np.pi*df.index.dayofyear/365),
    "weekday":  df.index.weekday
}, index=df.index)

full = pd.concat([df, cal], axis=1)

# 3) scale each column separately
scaler = StandardScaler()
X = scaler.fit_transform(full)
X = torch.tensor(X, dtype=torch.float32)          # (T, N + 3)

# 4) sliding-window dataset
HIST, HORIZON = 30, 7          # last 30 days → next 7   (modifiable)
seq, tgt = [], []
for t in range(HIST, len(X) - HORIZON + 1):
    seq.append(X[t-HIST:t])                # (30, N+3)
    tgt.append(X[t:t+HORIZON, :df.shape[1]])  # only sales part as target
seq  = torch.stack(seq)                    # (samples, 30, D)
tgt  = torch.stack(tgt)                    # (samples, 7, N)

class SparseGraphLearner(nn.Module):
    def __init__(self, n_nodes, l1_alpha=1e-3):
        super().__init__()
        self.A_logits = nn.Parameter(torch.randn(n_nodes, n_nodes))
        self.l1 = l1_alpha          # sparsity weight
    def forward(self):
        A = torch.sigmoid(self.A_logits)      # (0,1)
        A = A * (1 - torch.eye(A.size(0)))     # remove self-loops
        return A
    def l1_loss(self):
        return self.l1 * torch.abs(torch.sigmoid(self.A_logits)).sum()

##############
from torch_geometric.nn import GCNConv
class TGNN(nn.Module):
    def __init__(self, n_series, hidden=64, horizon=HORIZON):
        super().__init__()
        self.glearner = SparseGraphLearner(n_series)
        self.gc1 = GCNConv(in_channels=n_series+3, out_channels=hidden)
        self.tcn = nn.Conv1d(hidden, hidden, kernel_size=3, dilation=2, padding=2)
        self.head = nn.Linear(hidden, horizon)   # step-ahead per node
    def forward(self, seq):       # seq: (B, L, D)  where D=N+3
        A = self.glearner()       # (N,N)
        edge_index = A.nonzero().t()        # COO indices
        edge_weight = A[edge_index[0], edge_index[1]]

        # reshape for graph conv: treat every time step separately
        B, L, D = seq.shape
        x = seq.reshape(B*L, D)                  # nodes=batch*L
        x = self.gc1(x, edge_index, edge_weight)
        x = torch.relu(x).reshape(B, L, -1).permute(0,2,1)  # (B, hidden, L)

        h = torch.relu(self.tcn(x))              # temporal features
        h = torch.mean(h, dim=-1)                # (B, hidden)
        out = self.head(h)                       # (B, horizon)
        return out, A

###########################################################
#  Training loop (device-agnostic: CPU or GPU works alike) #
###########################################################
# ──────────────────────────────────────────────────────────
# Split seq / tgt time series samples into train / val / test
#     - chronological split to avoid data leakage
# ──────────────────────────────────────────────────────────
TOTAL = len(seq)
train_end = int(0.7 * TOTAL)            # 70% for training
val_end   = int(0.85 * TOTAL)           # 15% validation, 15% test

seq_train, tgt_train = seq[:train_end], tgt[:train_end]
seq_val,   tgt_val   = seq[train_end:val_end], tgt[train_end:val_end]
seq_test,  tgt_test  = seq[val_end:], tgt[val_end:]

print(f"train:{len(seq_train)}, val:{len(seq_val)}, test:{len(seq_test)}")

# ──────────────────────────────────────────────────────────
# Device setup + training loop
# ──────────────────────────────────────────────────────────
import torch, torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(">> training on", DEVICE)

model   = TGNN(n_series=df.shape[1]).to(DEVICE)
opt     = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = nn.L1Loss()                      # MAE in scaled space
BATCH   = 64

def batch_iter(x, y, bs):
    idx = torch.randperm(len(x))
    for i in range(0, len(x), bs):
        j   = idx[i:i+bs]
        yield x[j].to(DEVICE), y[j].to(DEVICE)

for epoch in range(80):
    # ---- training ----
    model.train(); train_mae = 0.0
    for xb, yb in batch_iter(seq_train, tgt_train, BATCH):
        opt.zero_grad()
        pred, _   = model(xb)
        loss      = loss_fn(pred, yb.mean(-1)) + model.glearner.l1_loss()
        loss.backward(); opt.step()
        train_mae += loss_fn(pred, yb.mean(-1)).item() * len(xb)

    # ---- validation ----
    model.eval(); val_mae = 0.0
    with torch.no_grad():
        for xb, yb in batch_iter(seq_val, tgt_val, BATCH):
            pred, _ = model(xb)
            val_mae += loss_fn(pred, yb.mean(-1)).item() * len(xb)

    train_mae /= len(seq_train)
    val_mae   /= len(seq_val)
    print(f"E{epoch:02d}  train-MAE {train_mae:.4f} | val-MAE {val_mae:.4f}")

#########
# ───────────────────────────────
# 1) Compute overall average scaling factor (a single scalar)
# ───────────────────────────────
avg_scale = torch.tensor(scaler.scale_[:df.shape[1]].mean(), dtype=torch.float32)
avg_mean  = torch.tensor(scaler.mean_ [:df.shape[1]].mean(),  dtype=torch.float32)

# ───────────────────────────────
# 2) Prediction & inverse transform
# ───────────────────────────────
model.eval(); preds, trues = [], []
with torch.no_grad():
    for xb, yb in batch_iter(seq_test, tgt_test, BATCH):
        p, _ = model(xb.to(DEVICE))
        preds.append(p.cpu())
        trues.append(yb.mean(-1).cpu())      # consistent with training

preds = torch.cat(preds) * avg_scale + avg_mean     # (samples, horizon)
trues = torch.cat(trues) * avg_scale + avg_mean

# ───────────────────────────────
# 3) Compute out-of-time MAPE
# ───────────────────────────────
mape = ((preds - trues).abs() / trues.clamp(min=1e-8)).mean().item() * 100
print(f"Out-of-time MAPE = {mape:.2f} %")

A = model.glearner().cpu().detach().numpy()
import networkx as nx, matplotlib.pyplot as plt
G = nx.from_numpy_array(A, create_using=nx.DiGraph)
# threshold for visibility
G = nx.DiGraph( (u,v,d) for u,v,d in G.edges(data=True) if d['weight']>0.15 )
nx.draw(G, node_size=300, arrows=True)

# =========================================================
# 5.  Classical baselines: SARIMAX & SES for comparison
# =========================================================
# -----------------------------------------------------------
# generic one-step-ahead rolling forecaster  (works for both)
# -----------------------------------------------------------
def rolling_forecast(model_builder, history, steps, fit_kwargs=None):
    fit_kwargs = fit_kwargs or {}
    hist  = list(history)
    out   = []
    for _ in range(steps):
        model   = model_builder(hist)          # build fresh model
        result  = model.fit(**fit_kwargs)
        fc      = result.forecast(1)[0]
        out.append(fc)
        hist.append(fc)                        # roll the window
    return np.array(out)

test_len  = len(seq_te)                        # number of rolling steps
train_raw = df.iloc[:val_end + HIST]           # data up to test start

sarimax_preds, ses_preds = [], []
for col in df.columns:
    trn = train_raw[col].values
    # SARIMAX(1,0,1)
    sarimax_preds.append(
        rolling_forecast(
            lambda h: SARIMAX(h, order=(1,0,1)),
            trn, test_len,
            fit_kwargs={"disp": False}
        )
    )
    # Simple Exponential Smoothing
    ses_preds.append(
        rolling_forecast(
            lambda h: SimpleExpSmoothing(h, initialization_method="estimated"),
            trn, test_len
        )
    )

sarimax_preds = np.column_stack(sarimax_preds).mean(axis=1)   # align shapes
ses_preds     = np.column_stack(ses_preds).mean(axis=1)

# ---------- MAPE helper ----------
def mape(pred, true):
    return np.mean(np.abs(pred-true) / np.clip(true, 1e-8, None)) * 100

truth_avg   = truth.mean(axis=1)           # (samples,)

sx_mape  = mape(sarimax_preds, truth_avg)
ses_mape = mape(ses_preds,     truth_avg)

print("\n=============  OUT-OF-TIME  MAPE  =============")
print(f"SARIMAX(1,0,1)  : {sx_mape:6.2f} %")
print(f"SES             : {ses_mape:6.2f} %")
