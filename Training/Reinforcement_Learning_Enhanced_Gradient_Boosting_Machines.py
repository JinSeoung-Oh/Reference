### From https://pub.towardsai.net/reinforcement-learning-enhanced-gradient-boosting-machines-77457e8cb4d9

"""
1. Overview
   This post demonstrates how reinforcement learning (RL) can directly enhance gradient boosting models (GBM) by dynamically adjusting 
   the learning rate at each boosting iteration. Rather than relying on traditional hyperparameter tuning or fixed learning rate schedules, 
   an RL agent is integrated into the boosting procedure to choose an optimal multiplier for the base learning rate. 
   This dynamic adjustment allows the GBM to better respond to evolving data patterns, leading to improved predictive performance.

   Experiments on both synthetic data and a Kaggle used-car pricing dataset validate the approach. 
   For regression tasks, the RL-enhanced GBM (RL-GBM) consistently outperforms competitive models like XGBoost, LightGBM, and Random Forest. 
   For classification tasks, the method achieves performance comparable to state-of-the-art models while outperforming simpler methods like AdaBoost.

2. Key Concepts and Mechanism
   -a. Gradient Boosting Recap
       Gradient boosting builds an ensemble of weak learners (typically decision trees) in a sequential manner. At iteration 𝑡:
       -1. An initial model 𝐹_0(𝑥) is defined (often a constant).
       -2. Residuals (pseudo-residuals) are computed:
           𝑟_(𝑖,𝑡)=𝑦_𝑖−𝐹_(𝑡−1)(𝑥_𝑖)
       -3. A new tree ℎ_𝑡(𝑥) is fitted to these residuals.
       -4. The ensemble is updated as:
           𝐹_𝑡(𝑥)=𝐹_(𝑡−1)(𝑥)+𝛼_𝑡⋅ℎ_𝑡(𝑥)
           where 𝛼_𝑡 is the learning rate. Typically, 𝛼_𝑡 might be determined via a line search.

   -b. Integrating Reinforcement Learning
       In the proposed method, rather than using a fixed learning rate 𝛼_𝑡 , an RL agent selects a multiplier 𝑚_𝑡 from a discrete set 
       (e.g., {08,0.9,1.0,1.1,1.2}). The effective learning rate becomes:
       𝛼_𝑡^(eff)=𝛼_𝑡×𝑚_𝑡
       The RL agent’s state is defined by a discretization of the current weighted training error, and its reward is computed as the decrease 
       in validation error from one boosting iteration to the next. Q-learning is used to update a Q-table, where the update rule is:
       𝑄(𝑠_𝑡,𝑎_𝑡)←𝑄(𝑠_𝑡,𝑎_𝑡)+𝛼_𝑞(𝑟_𝑡+𝛾max_𝑎𝑄(𝑠_(𝑡+1),𝑎)−𝑄(𝑠_𝑡,𝑎_𝑡))
       An epsilon-greedy policy is used to balance exploration and exploitation.

3. Detailed Implementation with Code
   The following Python code demonstrates the RL-enhanced GBM regression method. 
   It includes data preprocessing for a used-car price prediction task from a Kaggle dataset, the definition of metric functions,
   the RL-GBM regressor function, and comparisons with XGBoost, LightGBM, and Random Forest.
"""
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt

# === Load Data: car price: https://www.kaggle.com/competitions/playground-series-s4e9/data ===
df = pd.read_csv("train.csv")

# === Initial Cleanup ===
# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Drop rows where target (price) is missing or 0
df = df[df['price'].notnull() & (df['price'] > 0)]

# === Clean and Extract Numeric Info ===
def extract_hp(text):
    match = re.search(r"(\d+\.?\d*)\s*HP", str(text))
    return float(match.group(1)) if match else np.nan

def extract_engine_size(text):
    match = re.search(r"(\d+\.\d+)L", str(text))
    return float(match.group(1)) if match else np.nan

def extract_cylinder_count(text):
    match = re.search(r"(\d+)\s*[Vv]?\s*[Cc]ylinder", str(text))
    return int(match.group(1)) if match else np.nan

df['engine_hp'] = df['engine'].apply(extract_hp)
df['engine_L'] = df['engine'].apply(extract_engine_size)
df['cylinder'] = df['engine'].apply(extract_cylinder_count)

# Drop original engine column
df.drop(columns=['engine'], inplace=True)

# === Step 4: Handle Missing Values ===
# Add missing flags for selected columns
for col in ['int_col', 'transmission']:
    df[f'flag_{col}_missing'] = df[col].isnull().astype(int)

# Fill missing with placeholder
df['int_col'] = df['int_col'].fillna('Missing')
df['transmission'] = df['transmission'].fillna('Missing')

# Fill numeric engine values
for col in ['engine_hp', 'engine_L', 'cylinder']:
    df[col] = df[col].fillna(df[col].median())

# === Step 5: Categorical One-hot Encoding ===
cat_cols = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col', 'accident', 'clean_title']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# === Drop unnecessary or ID fields ===
df.drop(columns=['id'], inplace=True)

# === Correlation with Target (price) ===
target = 'price'
features = [col for col in df.columns if col != target]
corr_values = df[features].apply(lambda x: x.corr(df[target]))
abs_corr = corr_values.abs().sort_values(ascending=False)

# Select top N features
top_n = 20
top_features = abs_corr.head(top_n).index.tolist()

# === Final Model Data ===
model_df = df[top_features + [target]].dropna()

features = ['milage',  'model_year',  'engine_hp',
 'accident_None reported',  'brand_Lamborghini',
 'transmission_A/T',  'engine_L',  'cylinder',
 'brand_Bentley',  'brand_Porsche',
 'int_col_Nero Ade',  'transmission_7-Speed Automatic with Auto-Shift',
 'transmission_6-Speed A/T',
 'transmission_8-Speed Automatic with Auto-Shift',
 'int_col_Gray', 'int_col_Beige',  'transmission_8-Speed Automatic',
 'model_911 GT3',  'brand_Rolls-Royce', 
 'transmission_8-Speed A/T']

model_df['price'] = np.sqrt(model_df['price'] + 1)

print("Selected Features:\n", top_features)
print("\nCleaned Model DataFrame shape:", model_df.shape)

#############################################
# Metric Functions
#############################################
def mean_absolute_percentage_error(y_true, y_pred):
    eps = 1e-6
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + eps)) * 100.0

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

#############################################
# RL-GBM Regressor
#############################################
def rl_gbm_regressor(X_train, y_train, X_val, y_val, X_test, y_test,
                     base_learn_rate=0.1, M=200,
                     max_depth=3,
                     actions=[0.4, 0.6, 0.8, 0.9, 1.0, 1.1, 1.2]):
    n_train = len(X_train)
    n_features = X_train.shape[1]
    feature_names = X_train.columns

    num_states = 10
    num_actions = len(actions)
    Q = np.zeros((num_states, num_actions))
    epsilon = 0.1
    alpha_q = 0.1
    gamma_q = 0.9

    F_train = np.zeros(n_train)
    F_val   = np.zeros(len(X_val))
    F_test  = np.zeros(len(X_test))
    feature_importance_sum = np.zeros(n_features, dtype=np.float64)

    def get_state(g):
        avg_g = np.mean(np.abs(g))
        idx = int(avg_g / 10.0)
        return min(idx, num_states - 1)

    prev_val_mape = mean_absolute_percentage_error(y_val, F_val)
    train_mape_hist, val_mape_hist, test_mape_hist = [], [], []

    for t in range(M):
        g_train = y_train - F_train
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=1)
        tree.fit(X_train, g_train)
        tree_importance = tree.feature_importances_

        g_pred_train = tree.predict(X_train)
        g_pred_val   = tree.predict(X_val)
        g_pred_test  = tree.predict(X_test)

        state = get_state(g_train)
        action_idx = np.random.randint(num_actions) if np.random.rand() < epsilon else np.argmax(Q[state])
        alpha_eff = base_learn_rate * actions[action_idx]

        feature_importance_sum += alpha_eff * tree_importance

        F_train += alpha_eff * g_pred_train
        F_val   += alpha_eff * g_pred_val
        F_test  += alpha_eff * g_pred_test

        val_mape_now = mean_absolute_percentage_error(y_val, F_val)
        reward = prev_val_mape - val_mape_now
        prev_val_mape = val_mape_now

        next_state = get_state(y_train - F_train)
        Q[state, action_idx] += alpha_q * (reward + gamma_q * np.max(Q[next_state]) - Q[state, action_idx])

        train_mape_hist.append(mean_absolute_percentage_error(y_train, F_train))
        val_mape_hist.append(val_mape_now)
        test_mape_hist.append(mean_absolute_percentage_error(y_test, F_test))

    total = feature_importance_sum.sum()
    if total > 0:
        feature_importance_sum /= total

    feat_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'RLGBM_Importance': feature_importance_sum
    }).sort_values('RLGBM_Importance', ascending=False).reset_index(drop=True)

    print("\n=== RL-GBM Feature Importances ===")
    print(feat_importance_df)

    return F_test, Q, train_mape_hist, val_mape_hist, test_mape_hist

#############################################
# Load and Prepare Real Car Data
#############################################

# Feature columns and target

car_df = model_df.copy()
feature_cols = features[:]
target_col = 'price'

# Drop rows with missing values
car_df = car_df.dropna(subset=feature_cols + [target_col])

# Split data
X = car_df[feature_cols]
y = car_df[target_col]
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

#############################################
# Run RL-GBM
#############################################
final_test_preds, Q_table, train_mape_hist, val_mape_hist, test_mape_hist = rl_gbm_regressor(
    X_train, y_train, X_val, y_val, X_test, y_test,
    base_learn_rate=0.03, M=300, max_depth=4
)

final_test_preds, Q_table, train_mape_hist, val_mape_hist, test_mape_hist = rl_gbm_regressor(
    X_train, y_train, X_val, y_val, X_test, y_test,
    base_learn_rate=0.02, M=400, max_depth=5
)

# Metrics
mape_rl  = mean_absolute_percentage_error(y_test, final_test_preds)
rmse_rl  = rmse(y_test, final_test_preds)
print("\n=== RL-GBM Regressor Results ===")
print(f"MAPE(%): {mape_rl:.4f}, RMSE: {rmse_rl:.4f}")

#############################################
# Compare with XGB, LGB, RF
#############################################
xgb_reg = xgb.XGBRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, objective='reg:squarederror', random_state=42)
xgb_reg.fit(X_train, y_train)
xgb_preds = xgb_reg.predict(X_test)
xgb_mape  = mean_absolute_percentage_error(y_test, xgb_preds)
xgb_rmse_ = rmse(y_test, xgb_preds)

lgb_reg = lgb.LGBMRegressor(n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42)
lgb_reg.fit(X_train, y_train)
lgb_preds = lgb_reg.predict(X_test)
lgb_mape  = mean_absolute_percentage_error(y_test, lgb_preds)
lgb_rmse_ = rmse(y_test, lgb_preds)

rf_reg = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
rf_reg.fit(X_train, y_train)
rf_preds = rf_reg.predict(X_test)
rf_mape  = mean_absolute_percentage_error(y_test, rf_preds)
rf_rmse_ = rmse(y_test, rf_preds)

print("\n=== Performance Comparison ===")
print(f"RL-GBM => MAPE: {mape_rl:.4f}  RMSE: {rmse_rl:.4f}")
print(f"XGBoost => MAPE: {xgb_mape:.4f}  RMSE: {xgb_rmse_:.4f}")
print(f"LightGBM => MAPE: {lgb_mape:.4f}  RMSE: {lgb_rmse_:.4f}")
print(f"RandomForest => MAPE: {rf_mape:.4f}  RMSE: {rf_rmse_:.4f}")

# Plot MAPE over iterations
plt.figure(figsize=(8, 6))
plt.plot(train_mape_hist, label="Train MAPE", linewidth=2)
plt.plot(val_mape_hist, label="Val MAPE", linewidth=2)
plt.plot(test_mape_hist, label="Test MAPE", linewidth=2)
plt.xlabel("Boosting Iteration")
plt.ylabel("MAPE (%)")
plt.title("RL-GBM Regressor MAPE Over Iterations")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

"""
4. Explanation and Results
   -a. Data Preparation:
       The code loads the used-car dataset from Kaggle, performs initial cleanup (e.g., stripping whitespace, handling missing values), 
       extracts numeric features from textual engine descriptions, and applies one-hot encoding to categorical variables. 
       The top 20 features most correlated with the target (price) are selected.
   -b. Metric Functions:
       Two helper functions calculate the mean absolute percentage error (MAPE) and root mean squared error (RMSE).
   -c. RL-GBM Regressor:
       The function rl_gbm_regressor implements the RL-enhanced gradient boosting process. For each boosting iteration:
       -1. Residuals are computed.
       -2. A decision tree is trained on these residuals.
       -3. An RL agent (using a Q-table with an epsilon-greedy policy) selects a learning rate multiplier from a discrete set.
       -4. The effective learning rate scales the tree’s predictions, updating ensemble predictions for train, validation, and test sets.
       -5. The reward is defined as the improvement in validation error (MAPE reduction), and the Q-table is updated accordingly.
       -6. Feature importances are accumulated and normalized over iterations.
   -d. Model Evaluation:
       After training, the RL-GBM regressor is compared with XGBoost, LightGBM, and Random Forest. 
       MAPE and RMSE metrics are computed, and the evolution of MAPE over boosting iterations is plotted.
   -e. Results Summary:
       The printed output shows that the RL-GBM method achieves lower MAPE and RMSE compared to the benchmark models. For example:
       === Performance Comparison ===
       RL-GBM => MAPE: 18.9071  RMSE: 66.1632
       XGBoost => MAPE: 19.3608  RMSE: 66.3405
       LightGBM => MAPE: 19.3419  RMSE: 66.2819
       RandomForest => MAPE: 19.2944  RMSE: 66.8720
       Similar experiments with a second run and a case study on predicting customer purchase decisions further validate the approach, 
       with additional experiments extending the method to classification tasks.

5. Conclusion
   This work illustrates how integrating reinforcement learning into gradient boosting can dynamically optimize the learning rate at each iteration.
   By doing so, the RL-GBM method adapts to the current state of training errors, 
   leading to improved performance over conventional gradient boosting methods and standard ensemble models. 
   The detailed Python code provided demonstrates the full implementation from data preparation through model training, evaluation, and comparison,
   offering a practical pathway for building adaptive, high-performance predictive models with limited resources.
