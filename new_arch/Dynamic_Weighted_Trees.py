### From https://medium.com/data-science-collective/enhancing-time-series-forecasting-with-dynamic-weighted-trees-8dad9aeae112
### From https://github.com/datalev001/tree_TSM/tree/main

"""
1. Overview: A Weighted Decision Tree Framework for Time Series Forecasting
   The author presents a practical, interpretable, and high-performing decision tree model for forecasting time series, 
   combining the strengths of linear autoregression (AR) with nonlinear, rule-based decision tree structures.

   This framework integrates:
   -a. Autoregressive lags
   -b. Cyclical (seasonal) patterns
   -c. Exponential recency weighting
   -d. Moving average smoothing
   
   The result is a single-tree model (not an ensemble) that:
   -a. Adapts quickly to changing dynamics
   -b. Handles nonlinearities naturally
   -c. Maintains human interpretability
   -d. Avoids the complexity of black-box or ensemble approaches

2. Motivation for a Tree-Based Time Series Model
   Standard AR models like AR(1):
     yₜ = ϕ₁yₜ₋₁ + εₜ
     … assume linear relationships and equal weighting of past observations.

   Real-world time series, however:
   -a. Often have nonlinear behaviors
   -b. Exhibit recurring cycles (e.g., weekly, monthly)
   -c. Require adaptive weighting, giving priority to recent data

   A decision tree addresses these by:
   -a. Learning nonlinear threshold-based rules
   -b. Enabling splits like: "if yₜ₋₁ > 100, predict high; else, predict low"
   -c. Allowing seamless integration of custom features (lags, seasonality, weights)

3. Core Components of the Framework
   -a. Lagged Features 
       -1. Standard AR-style inputs: yₜ₋₁, yₜ₋₂, ..., yₜ₋ₖ
       -2. Capture short-term memory in the time series
   -b. Moving Average (MA)
       -1. Short-term smoothing of recent values:   MAₜ = (1/k) ∑_{i=0}^{k−1} yₜ₋ᵢ
       -2. Reduces noise and helps the tree generalize
   -c. Cyclical Features
       -1. Encode seasonal patterns (e.g., weekly, monthly) using sine/cosine transforms:   
           -1) sin_dow = sin(2π × day/7)
           -2) cos_dow = cos(2π × day/7)
       -2. Allows tree to recognize smooth seasonality without hard jumps (e.g., Sunday → Monday)

   -d. Exponential Decay Weighting
       -1. Gives more importance to recent observations:   wₜ = exp(−α(T_max − t)), where α > 0
       -2. Example:   If α = 0.05, t = 10 gets weight ≈ 1.0, t = 1 gets ≈ 0.64
       -3. Helps the tree focus more on recent trends or regime shifts

4. Tree Construction: Weighted Loss & Training
   -a. The tree minimizes a weighted least squares loss:
       L = ∑ᵢ wᵢ · (yᵢ − ŷᵢ)²
       Where:
       -1. xᵢ includes all features: lags, MA, sine/cosine seasonality
       -2. wᵢ is the exponential decay weight for that observation
       -3. ŷᵢ is the prediction from the tree
   -b. Splitting logic:
       -1. The tree finds feature-threshold splits that most reduce this weighted error
       -2. Each leaf predicts the weighted mean of its assigned y-values
   -c. Benefit:
       -1. Helps the model respond to recent changes without being distorted by older, now-irrelevant trends

5. Forecasting Procedure (Recursive)
   Once trained, forecasting proceeds in a recursive loop:
   -a. Prepare input:
       -1. Use the latest L observations to compute:
           -1) Lagged features
           -2) Moving average
           -3) Sine/cosine seasonality for forecast date
   -b. Run inference:
       -1. Feed this input vector to the trained tree
       -2. Traverse splits like:
           -1) “if sin_dow > 0.5 → left”
           -2) “if MA ≤ 120 → right”
       -3. Arrive at a leaf node
       -4. Return ŷₜ₊₁ as prediction
   -c. Iterate:
       -1. Append ŷₜ₊₁ to the history
       -2. Use it to generate input for forecasting ŷₜ₊₂, and so on

6. Interpretability & Insights
   Unlike ensemble or neural models, the decision tree is:
   -a. Transparent:
       -1. Each prediction path is a sequence of readable rules (e.g., “if MA ≤ 120 and cos_dow ≤ −0.3 → predict 130”)
   -b. Intuitive:
       -1. Rules reflect interpretable patterns:
           -1) Recency effects via decay
           -2) Seasonality via trigonometric encoding
           -3) Local patterns via lags or moving averages
   -c. Trustworthy for business use:
       -1. Decision-makers can audit forecasts and understand the “why” behind them

7. Performance Characteristics
   -a. Handles nonlinearities: Threshold splits capture interactions that linear AR misses
   -b. Responds to regime changes: Recency weighting prioritizes current data patterns
   -c. Noise-robust: MA smooths out short-term outliers
   -d. Seasonally aware: Sine/cosine inputs detect smooth periodicity
   The combined framework yields a robust yet interpretable forecasting solution that performs well even in volatile, 
   high-frequency, or shifting environments.

8. Summary
   -a. Integrative Strengths:
       -1. Lags: Learn short-term dependencies
       -2. MA: Reduce noise from spikes
       -3. Cycles: Model repeated patterns
       -4. Weighting: Emphasize recency
       -5. Trees: Learn conditional, nonlinear rules
   -b. Key Takeaways:
       -1. Embeds time-series knowledge (AR, MA, seasonality) into a single-tree model
       -2. Uses weighted squared-error minimization to favor recent training points
       -3. Produces interpretable decision rules that explain how forecasts are made
       -4. Offers a simple yet powerful tool — suitable for deployment or extension (e.g., boosting)
"""

import pandas as pd
import numpy as np
import math
import datetime
import re
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

# Import statsmodels for OLS estimation
import statsmodels.api as sm

# ----------------------
# Suppress warnings (if any)
# ----------------------
import warnings
warnings.filterwarnings("ignore")

# ======== Helper Functions for Tree Model =========

def add_cycle_features(df, date_col="DATE"):
    """
    Adds cyclical features based on day-of-week.
    """
    df[date_col] = pd.to_datetime(df[date_col])
    df['dayofweek'] = df[date_col].dt.dayofweek  # Monday=0, Sunday=6
    df['sin_dow'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['cos_dow'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    return df

def add_ma_feature(series, window=3):
    """
    Computes a moving-average feature over a given window.
    """
    return series.rolling(window=window, min_periods=1).mean()

def construct_lagged_dataset(df, target_col, L, decay_rate, date_col="DATE"):
    """
    For each time t (>= L) build feature vector composed of:
      - L lagged values (most recent first)
      - Current cyclical features (sin_dow, cos_dow)
      - MA feature: average of previous 3 observations
    Also computes an exponential decay weight.
    """
    df = df.copy().reset_index(drop=True)
    n = len(df)
    X, y, weights, dates = [], [], [], []
    T_max = n - 1
    for t in range(L, n):
        lag_feats = df[target_col].iloc[t-L:t].values[::-1].tolist()
        sin_dow = df.loc[t, 'sin_dow']
        cos_dow = df.loc[t, 'cos_dow']
        ma_feat = df[target_col].iloc[max(t-3, 0):t].mean()
        feat = lag_feats + [sin_dow, cos_dow, ma_feat]
        X.append(feat)
        y.append(df.loc[t, target_col])
        dates.append(df.loc[t, date_col])
        weight = math.exp(-decay_rate * (T_max - t))
        weights.append(weight)
    return np.array(X), np.array(y), np.array(weights), pd.to_datetime(dates)

def forecast_recursive(model, seed_window, seed_date, forecast_horizon, L, exog_df=None):
    """
    Given a fitted regression model (OLS or tree-based) and a seed window (most recent L observations),
    forecast recursively for forecast_horizon days.
    If exogenous features for future dates are provided in exog_df, they are used;
    otherwise, cycle features are computed from the forecast date.
    
    NOTE for tree-based models: the predicted feature vector must have the same dimension as during training.
    """
    forecasts = []
    current_window = list(seed_window)
    current_date = seed_date
    for h in range(forecast_horizon):
        next_date = current_date + pd.Timedelta(days=1)
        lag_feats = current_window[-L:][::-1]
        if exog_df is not None:
            cyc_vals = exog_df.loc[next_date]
            cyc_array = np.array([cyc_vals['sin_dow'], cyc_vals['cos_dow']])
        else:
            dow = next_date.dayofweek
            cyc_array = np.array([np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)])
        ma_feat = np.mean(current_window[-3:]) if len(current_window) >= 3 else np.mean(current_window)
        # Concatenate lag features, cycle features, and MA feature. Total length = L + 3.
        feat = np.concatenate([np.array(lag_feats), cyc_array, [ma_feat]])
        # For tree models, do not add constant.
        X_pred = feat.reshape(1, -1)
        y_hat = model.predict(X_pred)[0]
        forecasts.append(y_hat)
        current_window.append(y_hat)
        current_date = next_date
    return forecasts

def rolling_forecast_evaluation(model, df, target_col, L, forecast_horizon, decay_rate, start_idx):
    """
    Rolling-origin evaluation for the tree-based forecasting model.
    Returns MAPE (%) computed over all rolling origins.
    """
    errors = []
    n = len(df)
    for i in range(start_idx, n - forecast_horizon):
        if i - L < 0:
            continue
        seed_window = df[target_col].iloc[i-L:i].values
        seed_date = df.iloc[i-1]['DATE']
        preds = forecast_recursive(model, seed_window, seed_date, forecast_horizon, L)
        pred_avg = np.mean(preds)
        actual_avg = df[target_col].iloc[i:i+forecast_horizon].mean()
        if actual_avg == 0:
            continue
        errors.append(abs((actual_avg - pred_avg) / actual_avg))
    return np.mean(errors)*100 if errors else None

# ======== Additional Helper Functions: AR/ARMA with OLS =========

def rolling_forecast_evaluation_ar_ols(series, exog, p, forecast_horizon, start_idx):
    """
    Rolling-origin forecast evaluation for an AR(p) model using OLS.
    Predictors: constant, p lagged values, and exogenous cycle features.
    'series' is a pandas Series of the target.
    'exog' is a DataFrame (with same index as series) containing cycle features (e.g., sin_dow, cos_dow).
    Returns MAPE (%) over all rolling origins.
    """
    errors = []
    n = len(series)
    for i in range(start_idx, n - forecast_horizon):
        X_train, y_train = [], []
        for t in range(p, i):
            lags = series.iloc[t-p:t].values    # shape: (p,)
            cyc = exog.iloc[t].values             # shape: (2,)
            X_train.append(np.concatenate([lags, cyc]))  # total length = p + 2
            y_train.append(series.iloc[t])
        if len(X_train) == 0:
            continue
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        # Manually add constant column
        X_train_const = np.column_stack((np.ones(len(X_train)), X_train))
        model = sm.OLS(y_train, X_train_const).fit()
        seed_window = series.iloc[i-p:i].values.copy().tolist()
        current_index = i
        preds = []
        for k in range(forecast_horizon):
            cyc_forecast = exog.iloc[current_index].values  # shape: (2,)
            x_row = np.concatenate([np.array(seed_window[-p:]), cyc_forecast])  # length = p + 2
            X_pred = np.column_stack((np.ones(1), x_row.reshape(1, -1)))  # shape: (1, p+3)
            y_hat = model.predict(X_pred)[0]
            preds.append(y_hat)
            seed_window.append(y_hat)
            current_index += 1
            if current_index >= n:
                break
        if len(preds) < forecast_horizon:
            continue
        pred_avg = np.mean(preds)
        actual_avg = series.iloc[i:i+forecast_horizon].mean()
        if actual_avg == 0:
            continue
        errors.append(abs((actual_avg - pred_avg) / actual_avg))
    return np.mean(errors)*100 if errors else None

def rolling_forecast_evaluation_arma_ols(series, exog, forecast_horizon, start_idx):
    """
    Rolling-origin forecast evaluation for a simplified ARMA(1,1) model estimated via OLS.
    Predictors: constant, lag1 (y[t-1]), lagged error (approx. y[t-1] - y[t-2]),
                and exogenous cycle features.
    Returns MAPE (%) over all rolling origins.
    """
    errors = []
    n = len(series)
    for i in range(max(start_idx, 2), n - forecast_horizon):
        X_train, y_train = [], []
        for t in range(2, i):
            lag1 = series.iloc[t-1]
            lag_err = series.iloc[t-1] - series.iloc[t-2]
            cyc = exog.iloc[t].values
            X_train.append(np.concatenate([[lag1, lag_err], cyc]))  # length = 2+2 = 4
            y_train.append(series.iloc[t])
        if len(X_train) == 0:
            continue
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_train_const = np.column_stack((np.ones(len(X_train)), X_train))
        model = sm.OLS(y_train, X_train_const).fit()
        seed_y = series.iloc[i]
        seed_err = series.iloc[i] - series.iloc[i-1]
        preds = []
        current_index = i+1
        for k in range(forecast_horizon):
            cyc_forecast = exog.iloc[current_index].values
            x_row = np.concatenate([[seed_y, seed_err], cyc_forecast])  # length = 4
            X_pred = np.column_stack((np.ones(1), x_row.reshape(1, -1)))   # shape: (1, 5)
            y_hat = model.predict(X_pred)[0]
            preds.append(y_hat)
            new_err = y_hat - seed_y
            seed_y = y_hat
            seed_err = new_err
            current_index += 1
            if current_index >= n:
                break
        if len(preds) < forecast_horizon:
            continue
        pred_avg = np.mean(preds)
        actual_avg = series.iloc[i+1:i+1+forecast_horizon].mean()
        if actual_avg == 0:
            continue
        errors.append(abs((actual_avg - pred_avg) / actual_avg))
    return np.mean(errors)*100 if errors else None

# ======== Main Code: Data Preparation =========

# Read Electric Production data (expects columns: ['DATE','IPG2211A2N'])
df = pd.read_csv(r"C:\backupcgi\final_bak\Electric_Production_tm.csv")
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values(by='DATE').reset_index(drop=True)
df = add_cycle_features(df, date_col="DATE")
df['MA_3'] = add_ma_feature(df['IPG2211A2N'], window=3)

target_col = "IPG2211A2N"
date_col = "DATE"
decay_rate = 0.01
candidate_lags = [3, 5, 7, 9]
forecast_horizons = [3, 5, 7, 9]

# Split data into training and test sets (80% train, 20% test)
split_ratio = 0.8
split_idx = int(len(df) * split_ratio)
train_df = df.iloc[:split_idx].reset_index(drop=True)
test_df = df.iloc[split_idx:].reset_index(drop=True)

print("Total observations:", len(df))
print("Training observations:", len(train_df))
print("Test observations:", len(test_df))

# --- Tree-Based Forecasting (Using the above functions) ---
results_tree = {}
for horizon in forecast_horizons:
    best_mape = float('inf')
    best_L = None
    best_model = None
    print(f"\n[Tree Model] Forecast Horizon: {horizon} days")
    val_start_idx = int(0.8 * len(train_df))
    for L in candidate_lags:
        X_train, y_train, weights_train, _ = construct_lagged_dataset(train_df, target_col, L, decay_rate, date_col)
        tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
        tree_model.fit(X_train, y_train, sample_weight=weights_train)
        mape_val = rolling_forecast_evaluation(tree_model, train_df, target_col, L, horizon, decay_rate, start_idx=val_start_idx)
        if mape_val is None:
            continue
        print(f"  Candidate Lag L={L}: Validation MAPE = {mape_val:.2f}%")
        if mape_val < best_mape:
            best_mape = mape_val
            best_L = L
            best_model = tree_model
    print(f"--> Best Lag for horizon {horizon} days: L = {best_L} with Validation MAPE = {best_mape:.2f}%")
    X_train_full, y_train_full, weights_train_full, _ = construct_lagged_dataset(train_df, target_col, best_L, decay_rate, date_col)
    final_tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    final_tree_model.fit(X_train_full, y_train_full, sample_weight=weights_train_full)
    
    # --------------------------
    # Added Output: Print Decision Tree Rules
    # --------------------------
    from sklearn.tree import export_text
    feature_names = [f"lag_{i}" for i in range(1, best_L+1)] + ["sin_dow", "cos_dow", "ma_feat"]
    tree_rules = export_text(final_tree_model, feature_names=feature_names)
    print("Decision Tree Rules:")
    print(tree_rules)
    # --------------------------
    
    test_mape_tree = rolling_forecast_evaluation(final_tree_model, test_df, target_col, best_L, horizon, decay_rate, start_idx=best_L)
    print(f"[Tree Model] Test MAPE for horizon {horizon} days: {test_mape_tree:.2f}%")
    results_tree[horizon] = {'Best_Lag': best_L, 'Test_MAPE': test_mape_tree}

print("\nFinal Results for Tree-Based Model (Forecast Horizon: Best Lag, Test MAPE %):")
for h in forecast_horizons:
    res = results_tree[h]
    print(f"  {h} days -> Best Lag: {res['Best_Lag']}, Test MAPE: {res['Test_MAPE']:.2f}%")

# ======== Additional: OLS-based AR and ARMA Models =========

# Prepare test series and cycle exogenous regressors (sin_dow and cos_dow)
series_ar = test_df[target_col].reset_index(drop=True)
exog_ar = test_df[['sin_dow', 'cos_dow']].reset_index(drop=True)

print("\nEvaluating OLS-based AR and ARMA Models on Test Set:")

for horizon in forecast_horizons:
    # AR(3) replaces the previous AR(1) candidate; note start_idx is now 3.
    mape_ar3 = rolling_forecast_evaluation_ar_ols(series_ar, exog_ar, p=3, forecast_horizon=horizon, start_idx=3)
    # AR(2) remains.
    mape_ar2 = rolling_forecast_evaluation_ar_ols(series_ar, exog_ar, p=2, forecast_horizon=horizon, start_idx=2)
    # ARMA(1,1) remains.
    mape_arma11 = rolling_forecast_evaluation_arma_ols(series_ar, exog_ar, forecast_horizon=horizon, start_idx=2)
    
    print(f"\nForecast Horizon {horizon} days:")
    print(f"  AR(3) Test MAPE:    {mape_ar3:.2f}%")
    print(f"  AR(2) Test MAPE:    {mape_ar2:.2f}%")
    print(f"  ARMA(1,1) Test MAPE:{mape_arma11:.2f}%")
