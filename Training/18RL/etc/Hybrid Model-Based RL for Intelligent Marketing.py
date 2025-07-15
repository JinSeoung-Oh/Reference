### From https://pub.towardsai.net/hybrid-model-based-rl-for-intelligent-marketing-dyna-q-meets-transformer-models-and-bayesian-5d83b0ba7ea7

"""
0. Definition of Core Mechanism
   This system creates a digital lab loop: small real tests + large virtual simulations repeated iteratively to optimize coupon strategies. 
   It integrates:
   -a. Dyna-Q: Reinforcement learning for policy optimization
   -b. BG/NBD: Bayesian survival model predicting customer engagement
   -c. Transformer: Attention-based model predicting coupon profitability

1. Dyna-Q Overview
   -a. State s: A 5-tuple:
       (days_since, purchases_30d, avg_order_value, budget_pct, category_idx)
   -b. Action a: One of {0, 5, 10, 20} representing coupon value
   -c. Reward r: Net margin from issuing the coupon
   -d. Q-Update:
       -1. Q(s, a) ← Q(s, a) + α[r + γ·max_a′ Q(s′, a′) − Q(s, a)]
          - α ≈ 0.3 (decaying)
          - γ ≈ 0.95 (30-day horizon)
  -e. Simulated Learning: Dyna-Q can learn from synthetic transitions using models — BG/NBD and Transformer act as the simulation engine.

2. BG/NBD: Customer Survival Probability
   -a. Purpose: Estimate likelihood (P_alive) that a customer will return and engage
   -b. Inputs:
       -1. f: purchase frequency
       -2. r: recency
       -3. T: observation window
   -c. Output: A scalar P_alive ∈ [0, 1]
       Interpreted as a loyalty score
   -d. Role:
       -1. Ignores coupons, focuses on customer re-engagement probability
       -2. P_alive is added to state s and used in simulation reward

3. Transformer: Coupon Profit Estimator
   -a. Input: Category sequence over past 30 days + current trial coupon token
   -b. Mechanism:
       -1. Uses self-attention to find patterns (e.g., recent interest spikes)
       -2. Assigns attention weights to useful behavioral signals
   -c. Output: Predicted profit r^ from sending the coupon
   -d. Role:
       -1. Ignores survival but estimates monetary return
       -2. Complements BG/NBD by modeling immediate coupon effects

4. Combined Reward
   -a. For each (s, a), compute:
       r~ = P_alive × r^
   -b. Use:
       -1. Feed into synthetic simulations
       -2. Generate large amounts of replay data cheaply
       -3. Reduce real test risk and drop exploration rate ε to 0.05

5. Example Calculation
   -a. s = (days_since=7, avg_order_value=120, category=infant_food)
   -b. BG/NBD → P_alive = 0.82
   -c. Transformer → margin r^ = 105 − 10 = 95
   -d. Synthetic reward r~ = 0.82 × 95 ≈ 78
   -e. If Q(s,10)=65 and max_a′ Q(s′,a′)=70, new Q ≈ 74

6. System Architecture
   Component	| Role
   BG/NBD REST endpoint	| Returns P_alive
   Transformer CUDA service	| Returns r^
   CPU Dyna-Q worker	| Performs Q-updates and writes to Redis
   Nightly batch orchestrator	| Retrains models, simulates rollouts, outputs next-day coupons

7. System Value
   -a. BG/NBD: Filters out disengaged customers → reduces waste
   -b. Transformer: Accurately estimates expected revenue
   -c. Dyna-Q: Integrates both to compute best long-term coupon policy
   -d. Outcome:
       -1. Better ROI with lower cost
       -2. Rapid learning via simulation
       -3. Daily updates to marketing decisions
"""

import random, numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import trange

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lifetimes import BetaGeoFitter
from lifetimes.utils import calibration_and_holdout_data

import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 120

DATA_PATH = "repurchase_data.csv"   # <-- historical file

SEQ_LEN   = 30                                      # length of behaviour sequence
ACTION_SPACE = [0, 5, 10, 20]                       # coupon face values (¥)

# Load data --------------------------------------------------
df = pd.read_csv(DATA_PATH, parse_dates=["snapshot_date"])
coupon_map = {"none": 0, "5": 5, "10": 10, "20": 20}
df["coupon_value"] = df["coupon_offered"].map(coupon_map)
df["reward"]       = df["order_profit"] - df["coupon_value"]
df["category_idx"] = df["category"].astype("category").cat.codes

STATE_COLS = [
    "days_since_last_purchase", "purchases_last_30d",
    "avg_order_value",          "remaining_budget_pct",
    "category_idx"
]

# ----------------------------------------------------------------------
# Cold-start world model  ——  BG/NBD
# ----------------------------------------------------------------------
summary = calibration_and_holdout_data(
    transactions=df[["user_id","snapshot_date","order_placed"]],
    customer_id_col="user_id", datetime_col="snapshot_date",
    calibration_period_end="2025-05-31",
    observation_period_end="2025-06-30",
    freq="D"
)
bgf = BetaGeoFitter(penalizer_coef=1e-3)
bgf.fit(summary["frequency_cal"],
        summary["recency_cal"],
        summary["T_cal"])
print("BG/NBD parameters:", bgf.params_.round(3).to_dict())

# ----------------------------------------------------------------------
# Transformer world model
# ----------------------------------------------------------------------
class SeqDataset(Dataset):
    """Random 30-day category sequence  +  tabular features."""
    def __init__(self, frame: pd.DataFrame, seq_len: int = 30):
        self.df  = frame.reset_index(drop=True).copy()
        cats     = self.df["category_idx"].unique()
        self.df["seq"] = self.df["category_idx"].apply(
            lambda _ : np.random.choice(cats, seq_len)
        )
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row   = self.df.loc[idx]
        seq   = torch.as_tensor(row["seq"], dtype=torch.long)                # [L]
        feats = torch.as_tensor(row[STATE_COLS].to_numpy(np.float32),
                                dtype=torch.float32)                         # [F]
        act   = torch.tensor(row["coupon_value"] / 20.0, dtype=torch.float32)
        rew   = torch.tensor(row["reward"],         dtype=torch.float32)
        return seq, feats, act, rew

class WorldTransformer(nn.Module):
    """2-layer Attention encoder  +  MLP head."""
    def __init__(self, n_cat: int, d: int = 32):
        super().__init__()
        self.embed   = nn.Embedding(n_cat, d)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d, nhead=4,
                                       dim_feedforward=64,
                                       dropout=0.1,
                                       batch_first=True),
            num_layers=2)
        self.head = nn.Sequential(
            nn.Linear(d + len(STATE_COLS) + 1, 64),
            nn.ReLU(), nn.Linear(64, 1)
        )
    def forward(self, seq, feats, act):      # act shape [B]
        h = self.encoder(self.embed(seq))[:, 0, :]      # [CLS]
        x = torch.cat([h, feats, act.unsqueeze(1)], dim=1)
        return self.head(x).squeeze(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
wm     = WorldTransformer(df["category_idx"].nunique()).to(device)

ds     = SeqDataset(df, SEQ_LEN)
df["seq"] = ds.df["seq"]          # keep seq in main frame for Dyna-Q
loader = DataLoader(ds, batch_size=256, shuffle=True)
opt     = torch.optim.AdamW(wm.parameters(), lr=2e-3)
mse     = nn.MSELoss()

print("\nTraining Transformer world model (K epochs)…")
losses = []
for epoch in range(3):
    run = 0.0
    for seq, feat, act, rew in loader:
        seq, feat, act, rew = (t.to(device) for t in (seq, feat, act, rew))
        pred  = wm(seq, feat, act)
        loss  = mse(pred, rew)
        opt.zero_grad();  loss.backward();  opt.step()
        run += loss.item()
    losses.append(run / len(loader))
    print(f"  Epoch {epoch+1}: MSE = {losses[-1]:.3f}")

plt.plot(losses, marker="o"); plt.title("Transformer – training loss")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.tight_layout(); plt.show()

# ----------------------------------------------------------------------
# Roll-out helper  (BG/NBD  ×  Transformer)
# ----------------------------------------------------------------------
def roll_out(row_d: dict, action_val: int, model) -> float:
    """E[reward | s,a]  =  P(alive) · Transformer reward̂."""
    uid   = row_d["user_id"]
    alive = bgf.conditional_probability_alive(
        summary.loc[uid,"frequency_cal"] if uid in summary.index else 0,
        summary.loc[uid,"recency_cal"]   if uid in summary.index else 0,
        summary.loc[uid,"T_cal"]         if uid in summary.index else 1)

    seq   = torch.as_tensor(row_d["seq"], dtype=torch.long)[None].to(device)
    feats = torch.as_tensor([float(row_d[c]) for c in STATE_COLS],
                            dtype=torch.float32)[None].to(device)
    act   = torch.tensor([action_val/20.0], dtype=torch.float32,
                         device=device)                                  #  shape [1]
    model.eval()
    with torch.no_grad():
        r_hat = model(seq, feats, act).item()
    return alive * r_hat

# ----------------------------------------------------------------------
# Dyna-Q value iteration
# ----------------------------------------------------------------------
def bucket(row):
    """Discretise continuous state → small tuple."""
    return (int(row["days_since_last_purchase"] // 10),
            int(row["purchases_last_30d"]       // 2),
            int(row["avg_order_value"]          // 20),
            int(row["remaining_budget_pct"] * 10),
            int(row["category_idx"]))

gamma, alpha = 0.95, 0.30
eps0, eps_end = 0.30, 0.05
plan_k  = 20
epochs_rl = 4

Q      = defaultdict(float)
buffer = [(bucket(r), r["coupon_value"], r["reward"])
          for _, r in df.sample(2000, random_state=0).iterrows()]

print("\n=== Dyna-Q learning ===")
for ep in range(1, epochs_rl + 1):
    eps = max(eps_end, eps0 * (1 - ep/epochs_rl))
    today = df.sample(1500, random_state=ep)         # mock traffic
    for _, row in today.iterrows():
        s = bucket(row)
        a = random.choice(ACTION_SPACE) if random.random() < eps else \
            max(ACTION_SPACE, key=lambda act: Q[(s, act)])

        r_hat = roll_out(row.to_dict(), a, wm)       # ← pass model here
        best  = max(Q[(s, act)] for act in ACTION_SPACE)
        Q[(s, a)] += alpha * (r_hat + gamma*best - Q[(s, a)])

        buffer.append((s, a, r_hat))
        for _ in range(plan_k):                      # planning
            s2, a2, r2 = random.choice(buffer)
            best2 = max(Q[(s2, act)] for act in ACTION_SPACE)
            Q[(s2, a2)] += alpha * (r2 + gamma*best2 - Q[(s2, a2)])

    print(f"Epoch {ep}: ε={eps:.2f}, buffer={len(buffer):,}")

# ----------------------------------------------------------------------
# Recommendation list
# ----------------------------------------------------------------------
def recommend(row):
    s = bucket(row)
    # compare as floats
    best = max(ACTION_SPACE, key=lambda a: float(Q[(s, a)]))
    return best, round(float(Q[(s, best)]), 2)  

demo = df.sample(10, random_state=99)
records = [dict(user_id=int(r["user_id"]),
                days_since=int(r["days_since_last_purchase"]),
                rec_coupon=recommend(r)[0],
                exp_profit=recommend(r)[1])
           for _, r in demo.iterrows()]

pd.DataFrame(records).to_csv("recommendations.csv", index=False)
print("\n Sample recommendations:")
print(pd.DataFrame(records).head())

print("\n Pipeline complete – CSV is saved ")
