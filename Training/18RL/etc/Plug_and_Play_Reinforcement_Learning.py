### From https://pub.towardsai.net/plug-and-play-reinforcement-learning-for-real-time-forecast-recalibration-02c6203c429c
"""
1. The Problem with Traditional Time-Series Models
   An ARMA model trained on past prices, promotions, and holidays may perform well during initial validation
   but over time, things change:
   -a. Marketing campaigns shift
   -b. Competitor pricing drops
   -c. External factors (seasonality, events) disrupt sales

   The ARMA model becomes stale. But fully retraining the pipeline is:
   -a. Slow
   -b. Costly (compute + time)
   -c. Disruptive (breaks governance/alerting systems)
   -d. Wasteful (throws away hard-earned model structure)

2. The Solution: RL-Based Forecast Recalibration
   Instead of retraining, we preserve the ARMA model and add a small reinforcement learning agent 
   — like a smart "knob" — to gently adjust the forecast in real time.
   -a. Analogy:
       ARMA is your well-built luggage scale.
       The RL agent is a smart hand nudging the pointer based on weather (market) changes.
   -b. System Concept
       -a. Each day:
           -1. ARMA makes a forecast ŷ_t
           -2. The RL agent sees:
               -1) the ARMA residual (forecast error)
               -2) context (today’s price, promos, competitor price, etc.)
           -3. The agent applies a scaling factor: ŷ'_t = ŷ_t × (1 + δ_t) where δ_t ∈ [−0.2, 0.2]
           -4. After seeing the real sales y_t, the agent receives a **reward = −|y_t − ŷ'_t|`
           -5. The agent updates its behavior using Proximal Policy Optimization (PPO)

3. Components Breakdown
   -a. State s_t
       -1. Includes:
           -1) Latest ARMA forecast and residuals (last 7 days)
           -2) Own/competitor price
           -3) Promo/holiday flags
           -4) Sine/cosine for day-of-year
       ➡ 13-dimensional input summarizing the market + model behavior.
   -b. Action a_t
       -1. A correction factor δ in [−0.2, +0.2]
       -2. Applied multiplicatively: ŷ'_t = ŷ_t × (1 + δ)
   -c. Reward r_t
       -1. Negative absolute error: r_t = −|y_t − ŷ'_t|
       -2. Higher reward = lower error

4. PPO: Learning the Adjustment Policy
   PPO is used to train the RL agent. Key features:
   -a. Policy Representation
       -1. Outputs mean μ(s_t) and fixed standard deviation σ of a Gaussian
       -2. Correction factor is sampled, clipped to [0.8, 1.2]
   -b. Advantage Estimation
       -1. Uses Generalized Advantage Estimation (GAE):
           -1) Mixes short- and long-term feedback
           -2) λ = 0.95 spreads learning over ~5 days
   -c. Clipped Policy Loss
       -1. Ensures stability with a trust region:
           -1) Limits how much a policy can change in one step
           -2) Clip ratio: ρ_t = π_θ(a_t | s_t) / π_old(a_t | s_t)
           -3) Final loss uses min(ρ_t A_t, clip(ρ_t, 1−ε, 1+ε) × A_t)

5. Training Loop
   -a. Rollout: Simulate m days with current policy π_old, collect data.
   -b. Critic update: Learn value estimates to reduce variance.
   -c. Actor update: Apply PPO loss on mini-batches.
   -d. Sync: Replace π_old with the updated policy.
   This loop continues online — enabling real-time adaptation without retraining ARMA.

6. Case Study Behavior
   -a. In calm periods: RL tweaks are small, maintaining stability.
   -b. During promotions or external shocks: RL quickly adjusts forecasts to stay accurate.
   -c. The system automatically shifts between light-touch correction and aggressive adaptation.

7. Why This Works
   -a. Preserves ARMA’s structure and interpretability.
   -b. Avoids retraining — no broken pipelines, no revalidations.
   -c. Online, fast adaptation — reacts to market changes within hours.
   -d. Explainable policy — each correction can be traced to its context.
   -e. Safe updates — PPO ensures corrections don’t explode.

8. Grounding Key Concepts
   Symbol	| Meaning
   π_θ	| Current policy (probability distribution over actions)
   A_t	| Advantage: How much better the action was than expected
   λ	| GAE discount (0 = TD(0), 1 = Monte Carlo, 0.95 = balanced)
   ρ_t	| Importance ratio: how much policy has changed
   ε	| PPO clip range (default: 0.2)

9. Example:
   -a. ARMA underpredicts by −120 units
   -b. Agent scales by 1.08, new error = −10
   -c. Reward = −10; baseline expectation was +0.12
   -d. Advantage ≈ +0.85 → policy nudged toward similar future corrections

10. TRPO vs PPO
    -a. TRPO: Strong theoretical guarantees, but slow (second-order optimization)
    -b. PPO: Nearly as stable, much faster (clipping instead of KL constraint)
    In this system, PPO gives TRPO-level safety with SGD-level speed.

11. Takeaway
    This approach is a smart, lightweight patch over your trusted forecasting model.
    It:
       -a. Keeps models fresh without retraining
       -b. Is plug-and-play with ARMA (or others)
       -c. Adapts quickly in changing business environments
       -d. Brings RL robustness to classical time-series forecasting
    A practical bridge between statistical models and modern, adaptive machine learning.
"""

import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------------------------------------------------
#  DATA
# ---------------------------------------------------------
DATA_PATH  = "RL_sales_data.csv"

data = pd.read_csv(DATA_PATH, parse_dates=["date"]).set_index("date")

# ---------------------------------------------------------
#  FROZEN BASELINE   (train on first K days)
# ---------------------------------------------------------
TRAIN_END = 300
train_y = data["sales"][:TRAIN_END]
arma = ARIMA(train_y, order=(2,0,1)).fit()

baseline_pred, history = [], list(train_y)
for t in range(TRAIN_END, N):
    y_hat = ARIMA(history, order=(2,0,1)).fit().forecast()[0]
    baseline_pred.append(y_hat)
    history.append(data["sales"].iloc[t])

data.loc[data.index[TRAIN_END:], "baseline"] = baseline_pred

# ---------------------------------------------------------
#  GYM ENV  (fixed step() to avoid out-of-range)
# ---------------------------------------------------------
WINDOW, GAMMA = 7, 0.95
class SalesCorrectEnv(gym.Env):
    def __init__(self, df, start):
        super().__init__()
        self.df, self.start = df.reset_index(drop=True), start
        obs_dim = WINDOW + 4 + 2
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space      = spaces.Box(-1., 1., (1,), np.float32)
        self.reset()
    def _get_obs(self):
        res = self.residuals[-WINDOW:]
        if len(res) < WINDOW:
            res = np.concatenate([np.zeros(WINDOW-len(res)), res])
        row = self.df.loc[self.t]
        doy = 2*np.pi*(self.t % 365)/365
        return np.concatenate([res,
                               [row.price, row.comp_price, row.promo, row.mkt],
                               [np.sin(doy), np.cos(doy)]]).astype(np.float32)
    def step(self, action):
        w = np.clip(0.2*action[0]+1.0, 0.8, 1.2)
        row   = self.df.loc[self.t]
        y_hat = w * row.baseline
        r     = (abs(row.sales-row.baseline)-abs(row.sales-y_hat)) / (abs(row.sales-row.baseline)+1e-6)
        self.residuals.append(row.sales-y_hat)
        self.t += 1
        done   = self.t >= len(self.df)
        next_obs = np.zeros(self.observation_space.shape, np.float32) if done else self._get_obs()
        return next_obs, r, done, False, {}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.residuals = self.start, []
        return self._get_obs(), {}

env = DummyVecEnv([lambda: SalesCorrectEnv(data, TRAIN_END)])

# ---------------------------------------------------------
#  TRAIN PPO
# ---------------------------------------------------------
model = PPO("MlpPolicy", env,
            policy_kwargs=dict(net_arch=[64,64]),
            learning_rate=3e-4, batch_size=128, gamma=GAMMA,
            verbose=0).learn(30_000)

# ---------------------------------------------------------
# GENERATE RL-CORRECTED FORECASTS
# ---------------------------------------------------------
env_eval = SalesCorrectEnv(data, TRAIN_END); obs,_ = env_eval.reset()
corrected = []
for _ in range(TRAIN_END, N):
    act,_ = model.predict(obs, deterministic=True)
    obs, _, _, _, _ = env_eval.step(act)
    corrected.append(env_eval.residuals[-1] + data["baseline"].iloc[len(corrected)+TRAIN_END])
data.loc[data.index[TRAIN_END:], "corrected"] = corrected

# ---------------------------------------------------------
# METRICS & VISUAL
# ---------------------------------------------------------
test_y = data["sales"].iloc[TRAIN_END:]
base_p = data["baseline"].iloc[TRAIN_END:]
corr_p = data["corrected"].iloc[TRAIN_END:]

