### From https://medium.com/data-science-collective/residual-tail-boosting-a-fast-post-training-add-on-for-extra-accuracy-in-tree-ensembles-eac4e3c70058

"""
1. Residual-Tail Boosting (RTB): Summary
   Residual-Tail Boosting (RTB) is a flexible, lightweight extension to standard boosting models like XGBoost or LightGBM. 
   It adds a final corrective stage — a “tail learner” — to improve performance after the main “head” model has plateaued. 
   The tail specifically targets the residual errors remaining after the head has completed its own learning process.

2. Core Motivation
   -a. Traditional ensemble models stop improving when their built-in update rules can no longer extract new signal from the data.
   -b. RTB steps in after this plateau, applying a focused adjustment to the remaining errors using a small secondary learner.
   -c. This second-stage learner is fast, inexpensive, and modular — a “second pair of eyes” to catch overlooked patterns.

3. General Mechanism
   Let 𝐹^(0) (𝑥) be the head model’s output (log-odds for classification, raw prediction for regression), and let residuals be:
   -a. Classification: 𝑟^(𝑡)_𝑖 = 𝑦_𝑖 − 𝜎(𝐹^(𝑡) (𝑥_𝑖))
   -b. Regression: 𝑟^(𝑡)_𝑖 = 𝑦_𝑖 − 𝐹^(𝑡) (𝑥_𝑖)
   RTB operates in iterative rounds 𝑡=1,2,...,𝑇, applying a tail learner to current residuals, producing a delta 
   𝛿^(𝑡) (𝑥), which is scaled by step size 𝜈_𝑡, and used to update the model:
   𝐹^(𝑡) (𝑥) = 𝐹^(𝑡−1) (𝑥) + 𝜈_𝑡⋅𝛿^(𝑡) (𝑥)
   Each step uses a line-search over 𝜈_𝑡 to minimize held-out validation loss (e.g., log-loss or RMSE), 
   ensuring directly loss-driven updates.

4. Algorithmic Procedure
   -a. Split data into training (T) and validation (V).
   -b. Train head model on T to produce 𝐹^(0)
   -c. Iterate:
       -1. Compute residuals 𝑟^(𝑡)
       -2. Fit a tail model to 𝑟^(𝑡)
       -3. Predict deltas 𝛿^(𝑡) (𝑥)
       -4. Variance-scale 𝛿 so its magnitude is consistent
       -5. Perform a line search over 𝜈 to find best blend:
           𝜈_𝑡 = arg min_(𝜈∈{𝜈_1,...,𝜈_𝐾}) 𝐿(𝐹^(𝑡−1) + 𝜈⋅𝛿)
       -6. Update model if validation loss improves
       -7. Stop early if performance stagnates (stall counter)
   This loop dynamically and greedily improves validation performance using small, targeted updates.

5. Tail Variants
   -a. Ada-Tail (Binary Flag Stumps)
       -1. Converts signed residuals into binary targets:
           𝑦′_𝑖 = {1 if  𝑟_𝑖>0
                  {0 otherwise
       -2. Fits a weighted AdaBoost with a small number 𝑀 of stumps ℎ_𝑘, producing:
           𝛿^(𝑡) = ∑_𝑘 𝛼_𝑘 ℎ_𝑘(𝑥)
       -3. Delta is on log-odds scale, and variance-normalized before blending.
       -4. Captures coarse, directional error signals.
   -b. Residual-Tree Tail
       -1. Fits a shallow CART regressor to real-valued residuals.
       -2. Predicts:
                    Δ_𝑖 = 𝑇(𝑥_𝑖)
       -3. Captures non-binary, nuanced patterns the Ada-Tail might miss.

6. Why RTB Outperforms Traditional Blending
   -a. Dynamic residual correction:
       -1. Unlike static blending, which fixes weights 𝑤 and applies them once, RTB recalculates residuals after each delta.
   -b. Loss-guided and calibrated updates:
       -1. Each round optimizes true loss, not a surrogate.
       -2. Line search over 𝜈 avoids over/under-correction.
   -c. Model-agnostic:
       -1. Head can be trees, neural nets, etc.
       -2. Tail can be stumps, trees, k-NNs, tiny neural nets, etc.
   -d. Higher-order interaction modeling:
       -1. Tail updates adapt as residuals evolve, uncovering complex patterns in remaining error.

7. Theoretical and Practical Benefits
   -a. Modular:
       -1. Each delta 𝛿^(𝑡) is interpretable; contributions of individual tail rounds are isolated.
   -b. Low Overhead:
       -1. Tails require minimal compute; just a few shallow learners per round.
   -c. Early Stopping:
       -1. Automatically prevents overfitting by monitoring validation loss.
   -d. Plug-and-Play:
       -1. Universal loop works across head/tail types with no retraining of the head.
   -e. Proven Gains:
       -1. Consistent performance improvement on classification and regression benchmarks (log-loss, AUC, RMSE, MAPE), 
           without elaborate tuning.

8. Final Insight
   Residual-Tail Boosting (RTB) adds a final, corrective pass to any predictive model. Its compact, modular, and loss-driven design
   makes it a simple yet powerful tool for “last-mile” performance gains — 
   catching what the main model missed and pushing ensembles beyond their typical limits.
"""


import numpy as np
import pandas as pd
import warnings

##############data 1: give credit##########################
def getcorr_cut(Y, df_all, A_set, corr_threshold):
 
    import pandas as pd
    
    corr_list = []
    for feature in A_set:
        corr_value = abs(df_all[feature].corr(Y))
        if corr_value >= corr_threshold:
            corr_list.append({'varname': feature, 'abscorr': corr_value})
    df_corr = pd.DataFrame(corr_list)
    return df_corr

data = pd.read_csv('cs-training.csv')

def select_top_features_for_AB(df_all: pd.DataFrame,
                               A_set: list,
                               B_dummy_cols: list,
                               top_n_A: int,
                               top_n_B: int,
                               target_col: str = 'badflag',
                               corr_threshold: float = 0.0):
    """
    From DataFrame df_all (which contains the target, A_set columns, and dummy columns from B set),
    select the top_n_A features from A_set and top_n_B features from B_dummy_cols based on absolute
    correlation with the target (>= corr_threshold). Returns three lists: A_top, B_top, and the combined final_features.
    """
    Y = df_all[target_col]
    # Top features from A_set
    A_corr = getcorr_cut(Y, df_all, A_set, corr_threshold)
    A_corr = A_corr.sort_values('abscorr', ascending=False).head(top_n_A)
    A_top = A_corr['varname'].tolist()
    
    # Top features from B_dummy_cols
    B_corr = getcorr_cut(Y, df_all, B_dummy_cols, corr_threshold)
    B_corr = B_corr.sort_values('abscorr', ascending=False).head(top_n_B)
    B_top = B_corr['varname'].tolist()
    
    final_features = A_top + B_top
    return A_top, B_top, final_features

#############################################
# Load Data & Rename Columns
#############################################

# Rename long names to simple names
rename_map = {
    'SeriousDlqin2yrs': 'badflag',
    'RevolvingUtilizationOfUnsecuredLines': 'revol_util',
    'NumberOfTime30-59DaysPastDueNotWorse': 'pastdue_3059',
    'DebtRatio': 'debtratio',
    'MonthlyIncome': 'mincome',
    'NumberOfOpenCreditLinesAndLoans': 'opencredit',
    'NumberOfTimes90DaysLate': 'pastdue_90',
    'NumberRealEstateLoansOrLines': 'reloans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'pastdue_6089',
    'NumberOfDependents': 'numdep',
    'flag_MonthlyIncome': 'flag_mincome',
    'flag_NumberOfDependents': 'flag_numdep'
}
data = data.rename(columns=rename_map)

#############################################
# Fill Missing & Create A set
#############################################
missing_vars = ['mincome', 'numdep']
vars2= ['revol_util', 'debtratio']
for mv in missing_vars:
    flagcol = 'flag_' + mv
    if flagcol not in data.columns:
        data[flagcol] = data[mv].isnull().astype(int)
data[missing_vars] = data[missing_vars].fillna(data[missing_vars].mean())
A_set = missing_vars + ['flag_' + m for m in missing_vars] + vars2

#############################################
# Create B set by Dummies
#############################################
B_cats = ['pastdue_3059', 'pastdue_90',
          'reloans', 'pastdue_6089', 'opencredit']

dummy_frames = []
dummy_cols = []
for catvar in B_cats:
    if catvar not in data.columns:
        continue
    dums = pd.get_dummies(data[catvar], prefix=catvar)
    dummy_frames.append(dums)
    dummy_cols.extend(list(dums.columns))
B_dummies_df = pd.concat(dummy_frames, axis=1)

#############################################
#Feature Selection: Correlation and Top N from A & B
#############################################
tmp_df = pd.concat([data[['badflag']], data[A_set], B_dummies_df], axis=1)
A_top, B_top, final_features = select_top_features_for_AB(
    df_all=tmp_df,
    A_set=A_set,
    B_dummy_cols=list(B_dummies_df.columns),
    top_n_A=10,
    top_n_B=18,
    target_col='badflag',
    corr_threshold=0.002
)
print("Top A-set features:", A_top)
print("Top B-set features:", B_top)
print("Final combined feature list:", final_features)

#############################################
# Model DataFrame
#############################################
df_for_model = pd.concat([data[['badflag']], data[A_set], B_dummies_df], axis=1)
existing_feats = [f for f in final_features if f in df_for_model.columns]
model_df_credit = df_for_model[['badflag'] + existing_feats]

------------------------------------------------------------------------------------------------------------
 #### Ada‑tail Booster:
import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    log_loss, roc_auc_score, accuracy_score,
    mean_absolute_percentage_error, mean_squared_error
)
from scipy.stats import ks_2samp
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from xgboost.callback import EarlyStopping
import warnings
from sklearn.tree import DecisionTreeClassifier

# ============================================================
#  XGB “Head”  +  AdaBoost “Tail” for a Binary Target
#  Example run on the Give‑Me‑Some‑Credit data prepared above
# ============================================================

warnings.filterwarnings("ignore")

# -------- metrics helper --------
# ---------- metric helper ----------
def show_cls(y_true, proba, tag=""):
    print(f"{tag:<11} "
          f"LogLoss={log_loss(y_true, proba):.6f}  "
          f"AUC={roc_auc_score(y_true, proba):.4f}  "
          f"ACC={accuracy_score(y_true, proba >= 0.5):.4f}  "
          f"KS={ks_2samp(proba[y_true == 1], proba[y_true == 0]).statistic:.4f}")

# ---------- main routine ----------
def xgb_ada_tail(
    df, target, feats,
    xgb_rounds=50,
    ada_steps=30,
    nu_grid=np.linspace(0, 0.20, 11)
):
    """Run XGBoost head and AdaBoost tail; print validation metrics."""
    # -- split --
    X = df[feats].values
    y = df[target].values
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42
    )

    # -- XGB head --
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.05, max_depth=3,
        n_estimators=xgb_rounds, eval_metric="logloss",
        random_state=42, tree_method="hist", verbosity=0,
    )
    xgb_clf.fit(X_tr, y_tr)

    p_tr = xgb_clf.predict_proba(X_tr)[:, 1]
    p_vl = xgb_clf.predict_proba(X_vl)[:, 1]
    show_cls(y_vl, p_vl, "XGB‑head")

    # -- residuals --
    r_tr, r_vl = y_tr - p_tr, y_vl - p_vl
    z_tr = (r_tr > 0).astype(int)            # error direction
    w_tr = np.abs(r_tr) + 1e-6               # weight by magnitude

    # -- AdaBoost tail (scikit‑learn ≥1.2 uses `estimator=`) --
    ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=ada_steps, learning_rate=0.5,
        algorithm="SAMME.R", random_state=42,
    )
    ada.fit(X_tr, z_tr, sample_weight=w_tr)

    delta_tr = 2.0 * ada.decision_function(X_tr)
    delta_vl = 2.0 * ada.decision_function(X_vl)

    # variance scaling to match XGB margin scale
    scale = np.std(p_vl / (1 - p_vl + 1e-9)) / (np.std(delta_vl) + 1e-9)
    delta_tr *= scale
    delta_vl *= scale

    # -- line search for ν --
    margin_tr = np.log(p_tr / (1 - p_tr + 1e-9))
    margin_vl = np.log(p_vl / (1 - p_vl + 1e-9))
    best_loss, best_nu, best_p = 1e9, 0.0, p_vl

    for nu in nu_grid:
        p_tmp = 1 / (1 + np.exp(-(margin_vl + nu * delta_vl)))
        ll = log_loss(y_vl, p_tmp)
        if ll < best_loss:
            best_loss, best_nu, best_p = ll, nu, p_tmp

    # -- report --
    print(f"Chosen ν = {best_nu:.2f}")
    show_cls(y_vl, best_p, "XGB+Ada")
# ------------------------------------------------------------
#  Run on the prepared Give‑Me‑Some‑Credit dataframe
# ------------------------------------------------------------
target_col   = 'badflag'
feature_cols = existing_feats
xgb_ada_tail(model_df_credit, target_col, feature_cols)

----------------------------------------------------------------------------------

#### ADA Residual‑Tree Tail Booster
warnings.filterwarnings("ignore")

# ------------------------------------------------------------
#  Metric helper
# ------------------------------------------------------------
def show_cls(y, p, tag=""):
    print(f"{tag:<12}"
          f"LogLoss={log_loss(y,p):.6f}  "
          f"AUC={roc_auc_score(y,p):.4f}  "
          f"ACC={accuracy_score(y,p>=0.5):.4f}  "
          f"KS={ks_2samp(p[y==1],p[y==0]).statistic:.4f}")

# ------------------------------------------------------------
#  XGB head + residual‑tree tail
# ------------------------------------------------------------
def xgb_residual_tree_tail(
    df, target, feats,  xgb_rounds=50,
    tail_max_iter=30,  tail_patience=5,
    nu_grid=np.linspace(0, 0.30, 16),   # finer grid to 0.30
    tail_depth=2
):
    # ---- split ----
    X = df[feats].values
    y = df[target].values
    X_tr, X_vl, y_tr, y_vl = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42)

    # ---- XGB head ----
    xgb_clf = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.05, max_depth=3,
        n_estimators=xgb_rounds, eval_metric="logloss",
        random_state=42,tree_method="hist", verbosity=0,
    )
    xgb_clf.fit(X_tr, y_tr)

    p_tr = xgb_clf.predict_proba(X_tr)[:, 1]
    p_vl = xgb_clf.predict_proba(X_vl)[:, 1]
    show_cls(y_vl, p_vl, "XGB‑head")

    # margins
    margin_tr = np.log(p_tr / (1 - p_tr + 1e-9))
    margin_vl = np.log(p_vl / (1 - p_vl + 1e-9))

    best_loss = log_loss(y_vl, p_vl)
    stall = 0

    # ---- residual‑tree tail ----
    for t in range(1, tail_max_iter + 1):
        # signed residual (grad) and magnitude (weight)
        r_tr = y_tr - 1 / (1 + np.exp(-margin_tr))
        r_vl = y_vl - 1 / (1 + np.exp(-margin_vl))

        tree = DecisionTreeRegressor(max_depth=tail_depth, random_state=42)
        tree.fit(X_tr, r_tr)                         # weights not needed for CART-MSE

        delta_tr = tree.predict(X_tr)
        delta_vl = tree.predict(X_vl)

        # scale delta to same variance as margin
        scale = np.std(margin_vl) / (np.std(delta_vl) + 1e-9)
        delta_tr *= scale
        delta_vl *= scale

        # ---- line search over ν ----
        losses = []
        for nu in nu_grid:
            p_tmp = 1 / (1 + np.exp(-(margin_vl + nu * delta_vl)))
            losses.append(log_loss(y_vl, p_tmp))
        best_idx = int(np.argmin(losses))
        nu_star  = nu_grid[best_idx]
        new_loss = losses[best_idx]

        print(f"Round {t:02d}  depth={tail_depth}  ν={nu_star:.2f}  "
              f"valLoss={new_loss:.6f}")

        if new_loss >= best_loss - 1e-6:
            stall += 1
            if stall >= tail_patience:
                print("Early stop on residual‑tree tail\n")
                break
        else:
            stall = 0
            best_loss = new_loss
            margin_tr += nu_star * delta_tr
            margin_vl += nu_star * delta_vl

    # ---- final metrics ----
    p_final = 1 / (1 + np.exp(-margin_vl))
    show_cls(y_vl, p_final, "Final")

# ------------------------------------------------------------
# Call on Give‑Me‑Some‑Credit dataframe
# ------------------------------------------------------------
target_col   = 'badflag'
feature_cols = existing_feats          # from your preprocessing
xgb_residual_tree_tail(model_df_credit, target_col, feature_cols)
