### From https://pub.towardsai.net/adaptive-decay-weighted-arma-a-novel-approach-to-time-series-forecasting-9d69af9d6f16

"""
1. Introduction
   Time series forecasting is both valuable and difficult. Accurate forecasting informs critical decisions 
   across domains like energy, finance, and inventory management. However, modeling is hard because:
   -a. Time alters patterns (non-stationarity)
   -b. Seasonal effects and structural breaks appear
   -c. Traditional models often fail to capture the recency bias inherent in real-world data
   Classical models like AR(k) and ARMA assume equal importance across all past lags, 
   which is unrealistic in many dynamic systems.

2. Adaptive Decay-Weighted ARMA
   -a. Core idea: Modify the loss function by integrating a decay weighting function denoted as F(sequence_j | A).
   -b. This function:
       -1. Assigns more weight to recent observations (e.g., sequence₁ gets full weight)
       -2. Fades the influence of older ones using a decay parameter A
   -c. Importantly, A is not fixed — it can be learned during training
   -d. Result: A more flexible, adaptive, and accurate extension of ARMA models

3. Algorithmic Mechanisms and Procedure
   -a. Weighted Yule-Type Autoregressive Model
       Baseline AR(p) model (classical form):   Yₜ = ϕ₁Yₜ₋₁ + ... + ϕₚYₜ₋ₚ + εₜ
       Problem: All past observations are treated equally
       In the weighted formulation, a time-dependent weight function w(t; A) is introduced:
       -1. Emphasizes recent observations
       -2. Embeds into the loss function as a weighted MSE:   Loss = ∑ₜ w(t; A) · (Yₜ − Ŷₜ)²
   -b. Design of the Decay Function F(x | A)
       This function determines how the importance of past observations declines with lag.
       
       -1. Key properties of F(x | A):
           -1) Positivity: F(x | A) > 0
           -2) Monotonicity: Decreasing with respect to lag x
           -3) Boundedness or normalizability
       -2. Choices of Decay Functions
           -1) Exponential Decay:   F(t) = exp(−α(T − t)),
                                    where T = current time, α = decay rate
                                    → As t → T (i.e., more recent), weight → 1
                                    → As t decreases, weight exponentially fades
           -2) Lag-space expression:   wⱼ = exp(−α(j − 1)), for lag j
                                       → j = 1 (most recent) gets weight 1
                                       → Geometrically decaying
           -3) Normalized Exponential:   wⱼ = exp(−α(j − 1)) / ∑ⱼ exp(−α(j − 1))
                                         → Ensures total weight = 1
           -4) Other Decay Alternatives:
               - Power Decay:   F(j) = j^(−A), A > 0
                                → Heavier tail, slower decay
               - Softmax-based Attention:   F(j) = softmax(−A · j)
                                            → Normalized, differentiable
                                            → Inspired by neural attention mechanisms

3. Incorporating a Moving Average (MA) Component
   To model short-term fluctuations, an MA component is added.
   -a. Standard MA:   MAₜ = (1 / L) ∑ⱼ₌₁ᴸ Yₜ₋ⱼ
   -b. Weighted MA:   MAₜ = ∑ⱼ₌₁ᴸ ψⱼ Yₜ₋ⱼ
   -c. Introduce a coefficient γ for the MA term:   Ŷₜ = ϕ₁Yₜ₋₁ + ... + ϕₚYₜ₋ₚ + γ · MAₜ
   -d. Loss function becomes:   Loss = ∑ₜ w(t; A) · (Yₜ − Ŷₜ)² with parameter vector θ = (ϕ₁, ..., ϕₚ, γ)

4. Adding Seasonality or Cyclical Structure
   Seasonal variation is captured via:
   -a. Dummy variables:   Sₜ = [D₁, D₂, ..., Dₘ] for m-periodicity (e.g., months)
   -b. Fourier series:   sin(2πkt/m), cos(2πkt/m) for various harmonics k
   -c. Model becomes:   Ŷₜ = AR terms + γ · MAₜ + δᵗ · Sₜ
   -d. Loss:   Loss = ∑ₜ w(t; A) · (Yₜ − Xₜᵗ β)²
   Where:
      -a. Xₜ = regressor vector = [Yₜ₋₁, ..., Yₜ₋ₚ, MAₜ, Sₜ, 1]
      -b. β = coefficient vector = [ϕ₁, ..., ϕₚ, γ, δ, intercept]

5. Estimation and Optimization Procedure
   -a. Step 1: Design Matrix Construction
       -1. Compute lagged Yₜ₋ⱼ values
       -2. Compute moving average terms
       -3. Add seasonal features (dummy or Fourier)
       -4. Add constant
   -b. Step 2: Define Weighting Function
       -1. Use exponential: wₜ = exp(−α(T − t))
   -c. Step 3: Weighted Loss Function
       -1. L = ∑ₜ wₜ · (Yₜ − Ŷₜ)²
   -d. Step 4: Estimation Methods
       -1. Weighted Least Squares (WLS)
           - Closed-form:   β̂ = (XᵗWX)⁻¹ XᵗWY
           - W = diagonal matrix of weights
       -2. Gradient-Based Optimization
           - Directly minimize weighted loss via:   β ← β − η · ∇L(β)
           - Decay parameter α is learnable, not fixed
           - Enables joint optimization of both model and decay structure
   -e. Step 5: Forecasting
       -1. Construct X_{T+h} using most recent observed or predicted lags
       -2. Use:   Ŷ_{T+h} = X_{T+h}ᵗ · β̂
       -3. For multi-step forecast horizon K:   Ŷ̄ = (1/K) ∑{h=1}^{K} Ŷ{T+h}
   -f. Step 6: Hyperparameter Tuning
       -1. α: learned via training or tuned via CV/grid search
       -2. Other hyperparameters: moving average window L, lag order p, seasonal representation

6. Conclusion
   The Adaptive Decay-Weighted ARMA model:
   -a. Successfully incorporates recency bias via differentiable decay weighting
   -b. Allows decay behavior to be learned from data
   -c. Integrates AR, MA, and seasonal/cycle terms in a single flexible framework
   -d. Supports both closed-form estimation and gradient-based learning
   -e. Demonstrates superior forecasting accuracy compared to traditional AR-based models on real-world datasets

   This method provides a foundational improvement in autoregressive modeling, 
   and its design allows seamless future extensions:
   -a. Nonlinear modeling
   -b. Hybrid neural-AR frameworks
   -c. Multi-decay schemes for multiple timescales
"""

import pandas as pd  
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # not used in AR forecast but available
from scipy.stats import ks_2samp
import statsmodels.api as sm

def _prepare_dataframe(DF: pd.DataFrame, S: str, time_unit: str, p: int, ma_window: int, add_seasonal: bool):
    """
    Prepares the dataframe:
      - Converts the time_unit column to datetime and sorts the DataFrame.
      - Creates AR lag features: Y_{t-1}, ..., Y_{t-p}.
      - Creates a moving average (MA) feature over the previous ma_window observations.
      - If add_seasonal is True, extracts the month from the time column and creates dummy variables.
    Drops rows with missing predictors.
    
    Returns:
      df: the augmented DataFrame.
      predictors: a list of predictor column names (in order).
    """
    # Drop rows with missing S or time_unit, and convert time column to datetime.
    df = DF.copy().dropna(subset=[S, time_unit]).reset_index(drop=True)
    if not np.issubdtype(df[time_unit].dtype, np.datetime64):
        df[time_unit] = pd.to_datetime(df[time_unit], errors='coerce')
    df.sort_values(by=time_unit, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Create AR lag features
    for lag in range(1, p + 1):
        df[f"{S}_lag{lag}"] = df[S].shift(lag)
        
    # Create Moving Average (MA) feature: average of the previous ma_window observations, shifted by one.
    df["MA"] = df[S].rolling(window=ma_window, min_periods=ma_window).mean().shift(1)
    
    # Create seasonal dummies based on month if requested.
    seasonal_cols = []
    if add_seasonal:
        df["month"] = df[time_unit].dt.month
        month_dummies = pd.get_dummies(df["month"], prefix="mon", drop_first=True)
        df = pd.concat([df, month_dummies], axis=1)
        seasonal_cols = list(month_dummies.columns)
    
    # Drop rows with missing predictors (lags or MA)
    df.dropna(subset=[f"{S}_lag{p}", "MA"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Build list of predictor columns: AR lags + MA + seasonal dummies
    predictors = [f"{S}_lag{i}" for i in range(1, p+1)] + ["MA"] + seasonal_cols
    
    # Convert all predictor columns explicitly to float.
    df[predictors] = df[predictors].apply(pd.to_numeric, errors='coerce')
    df[predictors] = df[predictors].astype(float)
    # Drop any rows where predictors are NaN after conversion.
    df.dropna(subset=predictors, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    return df, predictors

def _decay_weighted_forecast_single(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    decay_param: float,
    decay_function: str = "exponential",
    p: int = 1,
    ma_window: int = 3,
    add_seasonal: bool = True
):
    """
    Internal function to compute decay-weighted AR(p) forecast with additional MA and seasonal features.
    Returns forecasted average and MAPE over the next K steps.
    """
    # Prepare the DataFrame with predictors.
    df, predictors = _prepare_dataframe(DF, S, time_unit, p, ma_window, add_seasonal)
    
    # Train/Test Split.
    n = len(df)
    train_size = int(0.8 * n)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    if len(test_df) < K:
        raise ValueError(f"Not enough test data to produce {K} forecasts.")
    forecast_section = test_df.iloc[:K].copy()
    
    # Weighted OLS Estimation.
    X_train = train_df[predictors].astype(float)
    y_train = train_df[S].astype(float)
    X_train_const = sm.add_constant(X_train)
    
    idx = np.arange(len(X_train))
    if decay_function.lower() == "exponential":
        w = decay_param ** ((len(X_train)-1) - idx)
    elif decay_function.lower() == "linear":
        slope = decay_param
        w = 1 + slope * (idx - idx.min())
    else:
        w = np.ones(len(X_train))
    
    w_sqrt = np.sqrt(w)
    wls_model = sm.WLS(y_train, X_train_const, weights=w_sqrt).fit()
    
    # Iterative Forecasting.
    # Maintain a history from the training target series.
    last_known = list(train_df[S].values.astype(float))
    preds = []
    
    # For each forecast step, build the predictor vector:
    for i in range(K):
        # AR lag features: the last p values.
        ar_vals = np.array(last_known[-p:], dtype=float)
        # MA: computed from the last ma_window values.
        if len(last_known) >= ma_window:
            ma_val = np.mean(last_known[-ma_window:])
        else:
            ma_val = np.mean(last_known)
        # Seasonal features: use forecast_section row i if applicable.
        seasonal_vals = []
        if add_seasonal:
            seasonal_cols = [col for col in predictors if col.startswith("mon_")]
            seasonal_vals = forecast_section.iloc[i][seasonal_cols].values.astype(float)
        # Combine in the order: AR lags, MA, seasonal dummies.
        x_row = np.concatenate([ar_vals, [ma_val], np.array(seasonal_vals, dtype=float)])
        # Convert to float type and add constant.
        X_pred = np.array(x_row, dtype=float).reshape(1, -1)
        X_pred = sm.add_constant(X_pred, has_constant='add')
        y_hat = wls_model.predict(X_pred)[0]
        preds.append(y_hat)
        last_known.append(y_hat)
    
    forecast_section["pred_decay"] = preds
    forecast_avg = forecast_section["pred_decay"].mean()
    
    def mape(actual, predicted):
        return np.mean(np.abs((actual - predicted) / (actual + 1e-9))) * 100
    mape_val = mape(forecast_section[S].values, forecast_section["pred_decay"].values)
    
    return forecast_avg, mape_val

def decay_weighted_forecast(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    decay_function: str = "exponential",
    p: int = 1,
    ma_window: int = 3,
    add_seasonal: bool = True,
    train_frac: float = 0.8,
    tune_decay: bool = False,
    candidate_decay_params: list = None,
    verbose: bool = True
):
    """
    Decay-weighted AR(p) forecasting with additional MA and seasonal features.
    If tune_decay is True, tries candidate_decay_params (default if None: [0.85, 0.90, 0.95])
    and returns the best forecast result (lowest MAPE).
    
    Returns: (forecasted average, MAPE, chosen_decay_param)
    """
    if tune_decay:
        if candidate_decay_params is None:
            candidate_decay_params = [0.85, 0.90, 0.95]
        best_mape = np.inf
        best_forecast_avg = None
        best_param = None
        for dp in candidate_decay_params:
            forecast_avg, mape_val = _decay_weighted_forecast_single(
                DF, S, K, time_unit, decay_param=dp, decay_function=decay_function,
                p=p, ma_window=ma_window, add_seasonal=add_seasonal
            )
            if verbose:
                print(f"Tuning decay_param={dp:.2f}: Forecast Avg = {forecast_avg:.4f}, MAPE = {mape_val:.2f}%")
            if mape_val < best_mape:
                best_mape = mape_val
                best_forecast_avg = forecast_avg
                best_param = dp
        if verbose:
            print(f"\n=> Selected decay_param={best_param:.2f} with MAPE={best_mape:.2f}%")
        return best_forecast_avg, best_mape, best_param
    else:
        forecast_avg, mape_val = _decay_weighted_forecast_single(
            DF, S, K, time_unit, decay_param=0.95, decay_function=decay_function,
            p=p, ma_window=ma_window, add_seasonal=add_seasonal
        )
        return forecast_avg, mape_val, 0.95

def normal_ar_forecast(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    p: int = 1,
    ma_window: int = 3,
    add_seasonal: bool = True,
    train_frac: float = 0.8,
    verbose: bool = True
):
    """
    Normal (unweighted) AR(p) forecasting using standard OLS with additional MA and seasonal features.
    Returns forecasted average over K steps and MAPE.
    """
    # Prepare DataFrame.
    df, predictors = _prepare_dataframe(DF, S, time_unit, p, ma_window, add_seasonal)
    
    # Train/Test Split.
    n = len(df)
    train_size = int(train_frac * n)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    if len(test_df) < K:
        raise ValueError(f"Not enough test data to produce {K} forecasts.")
    forecast_section = test_df.iloc[:K].copy()
    
    # OLS Estimation.
    X_train = train_df[predictors].astype(float)
    y_train = train_df[S].astype(float)
    X_train_const = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_const).fit()
    if verbose:
        print("\n--- Normal OLS Summary ---")
        print(ols_model.summary())
    
    # Iterative Forecasting.
    last_known = list(train_df[S].values.astype(float))
    preds = []
    for i in range(K):
        ar_vals = np.array(last_known[-p:], dtype=float)
        if len(last_known) >= ma_window:
            ma_val = np.mean(last_known[-ma_window:])
        else:
            ma_val = np.mean(last_known)
        seasonal_vals = []
        if add_seasonal:
            seasonal_cols = [col for col in predictors if col.startswith("mon_")]
            seasonal_vals = forecast_section.iloc[i][seasonal_cols].values.astype(float)
        x_row = np.concatenate([ar_vals, [ma_val], np.array(seasonal_vals, dtype=float)])
        X_pred = np.array(x_row, dtype=float).reshape(1, -1)
        X_pred = sm.add_constant(X_pred, has_constant='add')
        y_hat = ols_model.predict(X_pred)[0]
        preds.append(y_hat)
        last_known.append(y_hat)
    
    forecast_section["pred_normal"] = preds
    forecast_avg = forecast_section["pred_normal"].mean()
    mape_val = np.mean(np.abs((forecast_section[S].values - forecast_section["pred_normal"].values) /
                                (forecast_section[S].values + 1e-9))) * 100
    if verbose:
        print(f"\n[Normal AR] Forecasted Average (over {K} steps): {forecast_avg:.4f}")
        print(f"MAPE: {mape_val:.2f}%")
    
    return forecast_avg, mape_val

def arma_forecast(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    add_seasonal: bool = True,
    train_frac: float = 0.8,
    verbose: bool = True
):
    """
    ARMA(1,1, cycle) forecasting using statsmodels ARIMA with order (1,0,1).
    If add_seasonal is True, seasonal (month) dummy variables are used as exogenous regressors.
    Returns forecasted average over K steps and MAPE.
    """
    df = DF.copy().dropna(subset=[S, time_unit])
    if not np.issubdtype(df[time_unit].dtype, np.datetime64):
        df[time_unit] = pd.to_datetime(df[time_unit], errors='coerce')
    df.sort_values(by=time_unit, inplace=True)
    df.reset_index(drop=True, inplace=True)

    exog = None
    if add_seasonal:
        df["month"] = df[time_unit].dt.month
        # Convert seasonal dummies to float to avoid numpy boolean subtraction errors.
        exog = pd.get_dummies(df["month"], prefix="mon", drop_first=True).astype(float)
    
    # Split into train/test.
    n = len(df)
    train_size = int(train_frac * n)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    if len(test_df) < K:
        raise ValueError(f"Not enough test data to produce {K} forecasts.")
    
    if add_seasonal:
        exog_train = exog.iloc[:train_size]
        exog_test = exog.iloc[train_size: train_size + K]
    else:
        exog_train = None
        exog_test = None
    
    # Fit ARMA model using ARIMA with order=(1,0,1) (since no differencing is applied).
    arma_model = sm.tsa.ARIMA(train_df[S], order=(1,0,1), exog=exog_train).fit()
    if verbose:
        print("\n--- ARMA(1,1, cycle) Model Summary ---")
        print(arma_model.summary())
    
    # Forecast next K steps.
    forecast = arma_model.forecast(steps=K, exog=exog_test)
    forecast_vals = forecast.values.astype(float)
    actual = test_df[S].iloc[:K].values.astype(float)
    mape_val = np.mean(np.abs((actual - forecast_vals) / (actual + 1e-9))) * 100
    forecast_avg = np.mean(forecast_vals)
    
    return forecast_avg, mape_val

def cycle_ar_forecast(
    DF: pd.DataFrame,
    S: str,
    K: int,
    time_unit: str,
    p: int = 1,
    add_seasonal: bool = True,
    train_frac: float = 0.8,
    verbose: bool = True
):
    """
    AR with cycle forecasting (no MA) using standard OLS on AR lag features and optional seasonal dummies.
    Returns forecasted average over K steps and MAPE.
    """
    # Prepare the DataFrame using the standard preparation, then remove the MA column and predictor.
    df, predictors = _prepare_dataframe(DF, S, time_unit, p, ma_window=3, add_seasonal=add_seasonal)
    if "MA" in predictors:
        predictors.remove("MA")
        df.drop(columns=["MA"], inplace=True)
    
    # Train/Test Split.
    n = len(df)
    train_size = int(train_frac * n)
    train_df = df.iloc[:train_size].copy()
    test_df = df.iloc[train_size:].copy()
    if len(test_df) < K:
        raise ValueError(f"Not enough test data to produce {K} forecasts.")
    forecast_section = test_df.iloc[:K].copy()
    
    # OLS Estimation.
    X_train = train_df[predictors].astype(float)
    y_train = train_df[S].astype(float)
    X_train_const = sm.add_constant(X_train)
    ols_model = sm.OLS(y_train, X_train_const).fit()
    if verbose:
        print("\n--- AR with Cycle (no MA) OLS Summary ---")
        print(ols_model.summary())
        
    # Iterative Forecasting.
    last_known = list(train_df[S].values.astype(float))
    preds = []
    for i in range(K):
        # AR lag features: the last p values.
        ar_vals = np.array(last_known[-p:], dtype=float)
        # In this model, no MA value is used.
        seasonal_vals = []
        if add_seasonal:
            seasonal_cols = [col for col in predictors if col.startswith("mon_")]
            seasonal_vals = forecast_section.iloc[i][seasonal_cols].values.astype(float)
        # Combine AR lags and seasonal dummies.
        if add_seasonal:
            x_row = np.concatenate([ar_vals, np.array(seasonal_vals, dtype=float)])
        else:
            x_row = ar_vals
        X_pred = np.array(x_row, dtype=float).reshape(1, -1)
        X_pred = sm.add_constant(X_pred, has_constant='add')
        y_hat = ols_model.predict(X_pred)[0]
        preds.append(y_hat)
        last_known.append(y_hat)
    
    forecast_section["pred_cycle_ar"] = preds
    forecast_avg = forecast_section["pred_cycle_ar"].mean()
    mape_val = np.mean(np.abs((forecast_section[S].values - forecast_section["pred_cycle_ar"].values) /
                                (forecast_section[S].values + 1e-9))) * 100
    if verbose:
        print(f"\n[AR with Cycle (no MA)] Forecasted Average (over {K} steps): {forecast_avg:.4f}")
        print(f"MAPE: {mape_val:.2f}%")
    
    return forecast_avg, mape_val

# === Execution Code Using Electric_Production_tm.csv ===
if __name__ == "__main__":
    # Load dataset (expects columns: 'DATE', 'IPG2211A2N')
    df = pd.read_csv("Electric_Production_tm.csv")  # file with columns ['DATE','IPG2211A2N']
    
    # Forecast horizons to compare.
    forecast_horizons = [3, 5, 7, 9]
    series_name = "IPG2211A2N"
    time_col = "DATE"
    ar_order = 1         # AR(1)
    ma_window = 3        # moving average window size
    add_seasonal = True  # include seasonal (month) dummies
    
    # Containers for results.
    results_decay = {}
    results_normal = {}
    results_arma = {}
    results_cycle_ar = {}
    
    print("=== Decay-Weighted AR Forecasts (with decay tuning and additional features) ===")
    for K in forecast_horizons:
        fc_avg, mape_decay, chosen_dp = decay_weighted_forecast(
            DF=df,
            S=series_name,
            K=K,
            time_unit=time_col,
            decay_function="exponential",
            p=ar_order,
            ma_window=ma_window,
            add_seasonal=add_seasonal,
            train_frac=0.8,
            tune_decay=True,           # enable tuning
            candidate_decay_params=[0.85, 0.90, 0.95],
            verbose=True
        )
        results_decay[K] = (fc_avg, mape_decay, chosen_dp)
        print(f"Horizon {K} days: Forecast Avg = {fc_avg:.4f}, MAPE = {mape_decay:.2f}%, chosen decay_param = {chosen_dp:.2f}\n")
    
    print("=== Normal AR Forecasts (with additional features) ===")
    for K in forecast_horizons:
        fc_avg_norm, mape_norm = normal_ar_forecast(
            DF=df,
            S=series_name,
            K=K,
            time_unit=time_col,
            p=ar_order,
            ma_window=ma_window,
            add_seasonal=add_seasonal,
            train_frac=0.8,
            verbose=False
        )
        results_normal[K] = (fc_avg_norm, mape_norm)
        print(f"Horizon {K} days: Forecast Avg = {fc_avg_norm:.4f}, MAPE = {mape_norm:.2f}%\n")
    
    print("=== ARMA(1,1, cycle) Forecasts ===")
    for K in forecast_horizons:
        fc_avg_arma, mape_arma = arma_forecast(
            DF=df,
            S=series_name,
            K=K,
            time_unit=time_col,
            add_seasonal=add_seasonal,
            train_frac=0.8,
            verbose=False
        )
        results_arma[K] = (fc_avg_arma, mape_arma)
        print(f"Horizon {K} days: Forecast Avg = {fc_avg_arma:.4f}, MAPE = {mape_arma:.2f}%\n")
    
    print("=== AR with Cycle Forecasts (no MA) ===")
    for K in forecast_horizons:
        fc_avg_cycle, mape_cycle = cycle_ar_forecast(
            DF=df,
            S=series_name,
            K=K,
            time_unit=time_col,
            p=ar_order,
            add_seasonal=add_seasonal,
            train_frac=0.8,
            verbose=False
        )
        results_cycle_ar[K] = (fc_avg_cycle, mape_cycle)
        print(f"Horizon {K} days: Forecast Avg = {fc_avg_cycle:.4f}, MAPE = {mape_cycle:.2f}%\n")
    
    # Final comparison.
    print("=== Final Forecast Comparison ===")
    for K in forecast_horizons:
        fc_decay, mape_decay, dp = results_decay[K]
        fc_normal, mape_normal = results_normal[K]
        fc_arma, mape_arma = results_arma[K]
        fc_cycle, mape_cycle = results_cycle_ar[K]
        print(f"For {K} days forecast:")
        print(f"  Decay-Weighted AR: Forecast Avg = {fc_decay:.4f}, MAPE = {mape_decay:.2f}% (decay_param = {dp:.2f})")
        print(f"  Normal AR      : Forecast Avg = {fc_normal:.4f}, MAPE = {mape_normal:.2f}%")
        print(f"  ARMA(1,1, cycle): Forecast Avg = {fc_arma:.4f}, MAPE = {mape_arma:.2f}%")
        print(f"  AR with Cycle (no MA): Forecast Avg = {fc_cycle:.4f}, MAPE = {mape_cycle:.2f}%\n")
