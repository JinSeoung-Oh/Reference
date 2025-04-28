### From https://pub.towardsai.net/a-novel-and-practical-meta-booster-for-supervised-learning-eab9715a154a

"""
1. Concept and Positioning
   -a. Meta-Booster is a unified ensemble system that mixes the incremental updates (“deltas”) of several base learners at every boosting round.
   -b. It generalises boosting (sequential dependence) and stacking (static blending) by doing both simultaneously and iteratively.
   -c. Supported tasks: binary / multi-class classification and regression.

2. Three-Phase Cycle Per Boosting Round
   -a. Delta extraction
       -1. For each base learner 𝑚 train one extra step on the current training residuals.
       -2. Record its Δ𝑚^(𝑡):
           classification → margin increment (log-odds), regression → residual delta.
   -b. Stacked combination
       -1. Stack all 𝑀 delta columns into a skinny matrix 𝐷^(𝑡)∈𝑅^(𝑛×𝑀)
       -2. Solve a ridge-regularised least-squares problem on the held-out (validation) set
           min_𝑤∥𝑟^(𝑡)−𝐷^(𝑡)𝑤∥^2_2+𝜆∥𝑤∥^2_2
           to obtain weights 𝑤^(𝑡) (can be negative).
       -3. Forms a combined delta Δ^(𝑡)=𝐷^(𝑡)𝑤^(𝑡) 
   -c. Step-size line-search
       -1. Search a grid 𝜈∈{0.01,0.05,…,1.0} for 𝐿((𝑦^)^(𝑡)−𝜈Δ^(𝑡)) on the validation set.
       -2. Pick 𝜈∗  giving the lowest loss.
       -3. Update ensemble prediction (𝑦^)^(𝑡+1)=(𝑦^)^(𝑡)−𝜈∗Δ^(𝑡)
       -4. Early-stop if validation loss fails to improve for patience consecutive rounds.

3. Base-Learner Inventory
   Task	| Learner set	| Rationale
   Classification	| XGBoost, LightGBM, AdaBoost (shallow stumps), small NN	| Trees grab interactions; Ada fixes bias/variance; NN adds non-axis-aligned boundaries.
   Regression	| XGBoost, LightGBM, Elastic-net Linear, k-NN, small NN	| Linear anchors global trend; k-NN smooths local errors; trees + NN capture non-linearities.

4. Why Dynamic Stacking Beats Greedy Boosting
   -a. Greedy boosting often lets one strong model (e.g. XGBoost) dominate; weaker yet complementary learners rarely re-enter.
   -b. Least-squares stacking treats all deltas as a basis and projects the full gradient onto their span → steepest feasible descent.
   -c. Negative weights let, e.g., a linear model pull back tree over-shoots.
   -d. Line-search adapts the effective learning rate from ≈ 0.2 early (big residuals) to ≈ 0.02 near convergence.

5. Computation and Interpretability
   -a. Cost per round: one extra tree per tree-model, one stump for Ada, one gradient pass for NN, and an 𝑀×𝑀 linear solve (micro-seconds when 𝑀≤5)
   -b. On a 60 000-row dataset, 100 rounds finish in < 1 min on a laptop.
   -c. Tracking 𝑤^(𝑡) across rounds shows which learner contributes when — an interpretable “credit assignment” curve.

6. Extensibility
   -a. Replace the loss (e.g., binary log-loss → focal loss, MSE → pinball loss) without altering the three-phase loop.
   -b. Insert or freeze any learner (e.g., fine-tune only the NN under covariate shift).
   -c. Hyper-parameters {inventory,𝜆,𝜈-grid,patience} live in a single outer loop ⇒ clean cross-validation.

7. Bottom Line
   Meta-Booster is a modular, interpretable, validation-guarded ensemble template that:
   -a. Treats each learner’s micro-update as a basis vector.
   -b. Finds the best blend every round via ridge-regularised stacking.
   -c. Chooses an optimal step size by explicit line-search.
   The design generalises to arbitrary losses, data shifts, or learner inventories while requiring only a tiny linear solve per iteration
   — making it practical and adaptable for both research and production.
"""


##### Script on the Give Me Some Credit data for running the classification ###

import numpy as np
import pandas as pd
import warnings

##################corr function to choose top predictors################
def getcorr_cut(Y, df_all, A_set, corr_threshold):
    corr_list = []
    for feature in A_set:
        corr_value = abs(df_all[feature].corr(Y))
        if corr_value >= corr_threshold:
            corr_list.append({'varname': feature, 'abscorr': corr_value})
    df_corr = pd.DataFrame(corr_list)
    return df_corr

def select_top_features_for_AB(df_all: pd.DataFrame,
                               A_set: list,
                               B_dummy_cols: list,
                               top_n_A: int,
                               top_n_B: int,
                               target_col: str = 'badflag',
                               corr_threshold: float = 0.0):
  
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

data = pd.read_csv('cs-training.csv')

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
# Feature Selection: Correlation and Top N from A & B
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
# model data (give me credit) for classification test
#############################################
df_for_model = pd.concat([data[['badflag']], data[A_set], B_dummies_df], axis=1)
existing_feats = [f for f in final_features if f in df_for_model.columns]
model_df_credit = df_for_model[['badflag'] + existing_feats]

----------------------------------------------------------------------------------------------------------------------
#### preprocessing script
import numpy as np
import pandas as pd
import re
import warnings
import math

warnings.filterwarnings("ignore")
np.random.seed(42)

# --- 0. LOAD & PREPROCESS DATA ---
df = pd.read_csv("train.csv")
df.columns = df.columns.str.strip()
df = df[df.price.notnull() & (df.price > 0)]

def extract_hp(s):
    m = re.search(r"(\d+\.?\d*)\s*HP", str(s))
    return float(m.group(1)) if m else np.nan
def extract_L(s):
    m = re.search(r"(\d+\.\d+)L", str(s))
    return float(m.group(1)) if m else np.nan
def extract_cyl(s):
    m = re.search(r"(\d+)\s*[Vv]?[Cc]ylinder", str(s))
    return int(m.group(1)) if m else np.nan

df['engine_hp'] = df.engine.apply(extract_hp)
df['engine_L']  = df.engine.apply(extract_L)
df['cylinder']  = df.engine.apply(extract_cyl)
df.drop(columns='engine', inplace=True)
for c in ['int_col','transmission']:
    df[f'flag_{c}_missing'] = df[c].isnull().astype(int)
    df[c] = df[c].fillna('Missing')
for c in ['engine_hp','engine_L','cylinder']:
    df[c] = df[c].fillna(df[c].median())

cat_cols = ['brand','model','fuel_type','transmission',
            'ext_col','int_col','accident','clean_title']
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
model_df_carsprice = df.copy()  

df['target'] = (df.price > df.price.median()).astype(int)
df.drop(columns=['id','price'], inplace=True)

preds = [c for c in df if c!='target']
corrs = df[preds].apply(lambda col: abs(col.corr(df.target)))
df.drop(columns=corrs.nlargest(3).index, inplace=True)

preds2 = [c for c in df if c!='target']
top20 = df[preds2].apply(lambda col: abs(col.corr(df.target))).nlargest(20).index.tolist()
df = df[top20 + ['target']]

# model data (car price) for classiifcation test
model_df_cars = df.sample(frac=0.5, random_state=42).reset_index(drop=True)

# model data (car price) for regression test
model_df_carsprice = model_df_carsprice[top20 + ['price']]
model_df_carsprice['price'] = np.log(model_df_carsprice['price'] + 1)
model_df_carsprice = model_df_carsprice.sample(frac=0.5, random_state=42).reset_index(drop=True)

-------------------------------------------------------------------------------------------------------

### Meta‑Booster function
warnings.filterwarnings("ignore")

def run_meta_boost(model_df: pd.DataFrame, target: str, predictors: list, task: str = 'classification'):
    """
    Meta-boost ensemble for classification or regression.

    Parameters:
    - model_df: DataFrame with predictors and target.
    - target: target column name.
    - predictors: list of feature column names.
    - task: 'classification' or 'regression'.
    """
    # 1) Split data
    train_full, test_df = train_test_split(model_df, test_size=0.2, shuffle=False, random_state=42)
    train_df, val_df = train_test_split(train_full, test_size=0.2, shuffle=True, random_state=42)
    X_tr, y_tr = train_df[predictors].values, train_df[target].values
    X_vl, y_vl = val_df[predictors].values, val_df[target].values

    if task == 'classification':
        # Initialize base classifiers
        xgb_clf = xgb.XGBClassifier(n_estimators=30, use_label_encoder=False,
                                     eval_metric='logloss', random_state=42, verbosity=0)
        lgb_clf = lgb.LGBMClassifier(n_estimators=30, random_state=42, verbosity=-1)
        ada_clf = AdaBoostClassifier(n_estimators=30, random_state=42)
        nn_clf = MLPClassifier(hidden_layer_sizes=(20,), activation='tanh', solver='sgd',
                               learning_rate_init=0.01, max_iter=1, warm_start=True, random_state=42)
        # Fit base models
        xgb_clf.fit(X_tr, y_tr)
        lgb_clf.fit(X_tr, y_tr)
        ada_clf.fit(X_tr, y_tr)
        nn_clf.partial_fit(X_tr, y_tr, classes=np.unique(y_tr))

        # Initial held performance
        preds_vl = {
            'XGB': xgb_clf.predict_proba(X_vl)[:,1],
            'LGB': lgb_clf.predict_proba(X_vl)[:,1],
            'ADA': ada_clf.predict_proba(X_vl)[:,1],
            'NN':  nn_clf.predict_proba(X_vl)[:,1]
        }
        print("\nInitial held performance (classification):")
        for name, p in preds_vl.items():
            ll = log_loss(y_vl, p)
            auc = roc_auc_score(y_vl, p)
            acc = accuracy_score(y_vl, p >= 0.5)
            ks = ks_2samp(p[y_vl == 1], p[y_vl == 0]).statistic
            print(f" {name}: LogLoss={ll:.6f}, AUC={auc:.4f}, ACC={acc:.4f}, KS={ks:.4f}")

        # Initialize margins
        eps = 1e-9
        def logodds(p): return np.log((p + eps) / (1 - p + eps))
        def sigmoid(F): return 1 / (1 + np.exp(-F))

        # Choose best initial model by log-loss
        init = min(preds_vl, key=lambda n: log_loss(y_vl, preds_vl[n]))
        print(f"\nInitial margin chosen: {init}")
        # Compute initial F_tr and F_vl as log-odds
        prob_tr_init = {
            'XGB': xgb_clf.predict_proba(X_tr)[:,1],
            'LGB': lgb_clf.predict_proba(X_tr)[:,1],
            'ADA': ada_clf.predict_proba(X_tr)[:,1],
            'NN':  nn_clf.predict_proba(X_tr)[:,1]
        }[init]
        F_tr = logodds(prob_tr_init)
        F_vl = logodds(preds_vl[init])

        # Meta-boost settings
        best_loss = log_loss(y_vl, sigmoid(F_vl))
        nu_candidates = np.linspace(0, 1, 11)
        stall, max_rounds, patience = 0, 100, 5

        print("\nMeta-boosting classification (stacking & LR search):")
        for t in range(1, max_rounds + 1):
            # 1) Compute delta matrices
            deltas_tr, deltas_vl = [], []
            for name, clf in [('XGB', xgb_clf), ('LGB', lgb_clf),
                              ('ADA', ada_clf), ('NN', nn_clf)]:
                if name == 'XGB':
                    dtr = xgb.DMatrix(X_tr, label=y_tr, base_margin=F_tr)
                    bst = xgb.train(xgb_clf.get_xgb_params(), dtr, num_boost_round=1, verbose_eval=False)
                    new_tr = bst.predict(dtr, output_margin=True)
                    new_vl = bst.predict(xgb.DMatrix(X_vl, base_margin=F_vl), output_margin=True)
                elif name == 'LGB':
                    ds = lgb.Dataset(X_tr, label=y_tr, init_score=F_tr)
                    bst = lgb.train({'objective': 'binary', 'verbosity': -1}, ds, num_boost_round=1)
                    new_tr = bst.predict(X_tr, raw_score=True)
                    new_vl = bst.predict(X_vl, raw_score=True)
                elif name == 'ADA':
                    ada_clf.n_estimators += 1
                    ada_clf.fit(X_tr, y_tr)
                    new_tr = ada_clf.decision_function(X_tr)
                    new_vl = ada_clf.decision_function(X_vl)
                else:  # NN
                    nn_clf.partial_fit(X_tr, y_tr)
                    new_tr = logodds(nn_clf.predict_proba(X_tr)[:,1])
                    new_vl = logodds(nn_clf.predict_proba(X_vl)[:,1])
                deltas_tr.append(new_tr - F_tr)
                deltas_vl.append(new_vl - F_vl)

            D_tr = np.column_stack(deltas_tr)
            D_vl = np.column_stack(deltas_vl)

            # 2) Stacking: least-squares on probability residuals
            resid = y_vl - sigmoid(F_vl)
            w, *_ = np.linalg.lstsq(D_vl, resid, rcond=None)

            # 3) Combined deltas
            combo_tr = D_tr.dot(w)
            combo_vl = D_vl.dot(w)

            # 4) Line-search nu
            losses = []
            for nu in nu_candidates:
                p = sigmoid(F_vl + nu * combo_vl)
                losses.append(log_loss(y_vl, p))
            idx = int(np.argmin(losses))
            nu_best, new_loss = nu_candidates[idx], losses[idx]

            print(f" Round {t:2d}: loss_min={new_loss:.6f}, nu={nu_best:.2f}, weights={w}")

            # 5) Update or early stop
            if new_loss >= best_loss - 1e-6:
                stall += 1
                if stall >= patience:
                    print(" Early stopping\n")
                    break
            else:
                stall, best_loss = 0, new_loss
                F_tr += nu_best * combo_tr
                F_vl += nu_best * combo_vl

        # Final metrics
        p_meta = sigmoid(F_vl)
        print("\nFinal held performance (classification):")
        print(f" Meta-Boost: LogLoss={log_loss(y_vl,p_meta):.6f}, "
              f"AUC={roc_auc_score(y_vl,p_meta):.4f}, "
              f"ACC={accuracy_score(y_vl,p_meta>=0.5):.4f}, "
              f"KS={ks_2samp(p_meta[y_vl==1],p_meta[y_vl==0]).statistic:.4f}")
        return

    # --- regression branch unchanged ---
    base_models = {
        'XGB': xgb.XGBRegressor(n_estimators=30, random_state=42, verbosity=0),
        'LGB': lgb.LGBMRegressor(n_estimators=30, random_state=42, verbosity=-1),
        'NN': MLPRegressor(
            hidden_layer_sizes=(50, 20, 5),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=200,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10
        ),
        'KNN': KNeighborsRegressor(n_neighbors=5),
        'LIN': LinearRegression()
    }
    for m in base_models.values(): m.fit(X_tr, y_tr)

    preds_vl = {name: m.predict(X_vl) for name, m in base_models.items()}
    print("\nInitial held performance (regression):")
    for name, pred in preds_vl.items():
        mape = mean_absolute_percentage_error(y_vl, pred) * 100
        rmse = mean_squared_error(y_vl, pred, squared=False)
        print(f" {name}: MAPE={mape:.2f}%, RMSE={rmse:.6f}")

    init = min(preds_vl, key=lambda n: mean_squared_error(y_vl, preds_vl[n], squared=False))
    print(f"\nInitial prediction chosen: {init} based on RMSE")
    F_tr = base_models[init].predict(X_tr)
    F_vl = preds_vl[init].copy()
    best_loss = mean_squared_error(y_vl, F_vl, squared=False)
    nu = 0.06
    stall, max_rounds, patience = 0, 100, 5

    print("\nMeta-boosting regression (stacking & LR search):")
    for t in range(1, max_rounds+1):
        deltas_tr = np.column_stack([m.predict(X_tr) - F_tr for m in base_models.values()])
        deltas_vl = np.column_stack([m.predict(X_vl) - F_vl for m in base_models.values()])
        w, *_ = np.linalg.lstsq(deltas_vl, y_vl - F_vl, rcond=None)
        combo_tr = deltas_tr.dot(w)
        combo_vl = deltas_vl.dot(w)
        nu_candidates = np.linspace(0, 1, 11)
        losses = [mean_squared_error(y_vl, F_vl + nu_c * combo_vl, squared=False) for nu_c in nu_candidates]
        idx = int(np.argmin(losses)); nu_best = nu_candidates[idx]; new_loss = losses[idx]
        print(f" Round {t:2d}: losses={losses[:5]} | nu={nu_best:.2f}, new_loss={new_loss:.6f}")
        if new_loss >= best_loss - 1e-6:
            stall += 1
            if stall >= patience:
                print(" Early stopping\n")
                break
        else:
            stall, best_loss = 0, new_loss
            F_tr += nu_best * combo_tr
            F_vl += nu_best * combo_vl

    print("\nFinal held performance (regression):")
    final_mape = mean_absolute_percentage_error(y_vl, F_vl) * 100
    final_rmse = mean_squared_error(y_vl, F_vl, squared=False)
    print(f" Meta-Boost: MAPE={final_mape:.2f}%, RMSE={final_rmse:.6f}\n")
    print("Independent held performance (regression):")
    for name, m in base_models.items():
        pred = m.predict(X_vl)
        mape_i = mean_absolute_percentage_error(y_vl, pred) * 100
        rmse_i = mean_squared_error(y_vl, pred, squared=False)
        print(f" {name}: MAPE={mape_i:.2f}%, RMSE={rmse_i:.6f}")

## 1) carprice: classification###
run_meta_boost(model_df_cars, 'target', predictors = top20, task='classification')

## 2) Give me credit: classification#
run_meta_boost(model_df_credit, target='badflag', predictors=existing_feats)

## 3) carprice: regression ###
run_meta_boost(model_df_carsprice, target='price', predictors=top20, task='regression')
