### From https://pub.towardsai.net/distill-then-detect-a-practical-framework-for-error-aware-machine-learning-0e413d6fc7cc

"""
* Introduction & Motivation
  Real-world models often suffer from “big misses”—a small but critical subset (∼10%) of validation cases
  where predictions fail badly due to outliers, rare feature combinations, or patterns unseen during training.
  Such failures carry high costs (e.g., misclassifying risky credit applicants or failing to flag machines
  about to fail). 
  The proposed Error-Aware Distillation and Detection pipeline enables both 
  (1) diagnosing why the model fails on these hard cases and 
  (2) proactively flagging similar cases in future data for human review.

1. Quantifying Teacher Uncertainty via Entropy
   -a. Teacher model 
       𝑇 outputs a probability 𝑇(𝑥) for the positive class.
   -b. Entropy
       𝐻(𝑥)=−[𝑇(𝑥)ln𝑇(𝑥)+(1−𝑇(𝑥))ln(1−𝑇(𝑥))]
       -1. Peaks at ln2 for 𝑇(𝑥)=0.5 (maximal uncertainty ≈0.69 bits).
       -2. Approaches 0 for confident predictions (e.g. 95% confidence → ≈0.10 bits).
   -c. First signal: any 𝑥 with 𝐻(𝑥) above a chosen threshold lies in the teacher’s “blind zone.”

2. Teacher→Student Knowledge Distillation
   -a. Soft targets blend ground truth and teacher probability:
       𝑦′_𝑖=𝛼𝑦_𝑖+(1−𝛼)𝑇(𝑥_𝑖), 𝛼∈(0,1)
       -1. E.g. 𝛼=0.7, 𝑦_𝑖=1, 𝑇(𝑥_𝑖)=0.8 → 𝑦′_𝑖=0.7⋅1+0.3⋅0.8=0.94
   -b. Student loss combines cross-entropy and KL divergence:
       𝐿=∑_𝑖[(1−𝛼)ℓ_CE(𝑆(𝑥_𝑖),𝑦_𝑖)+𝛼KL(𝑦′_𝑖∥𝑆(𝑥_𝑖))]
   -c. Benefit: Student 𝑆 retains most teacher accuracy but generalizes better on easy cases and runs faster.

3. Meta-Model Gating of Teacher Errors
   -a. Residual label for each validation point:
       𝑒_𝑖=1[𝑇(𝑥_𝑖)misclassifies 𝑥_𝑖]∈{0,1}
   -b. Train meta-model 𝑀 on {(𝑥_𝑖,𝑒_𝑖)} to predict
       𝑀(𝑥)=Pr(𝑒=1∣𝑥),
       i.e. the probability that the teacher will err on input 𝑥
   -c. Role: 𝑀 learns “hard” regions—extreme values or rare feature combinations.

4. Unified Risk Scoring & Thresholding
   -a. Normalized teacher entropy: 𝐻(𝑥)/ln2∈[0,1]
   -b. Risk score:
       𝑅(𝑥)=𝑤⋅(𝐻(𝑥)/ln2)+(1−𝑤)⋅𝑀(𝑥), 𝑤∈[0,1]
       -1. With 𝑤=0.5, if 𝐻(𝑥)=0.4ln2(≈0.28 normalized) and 𝑀(𝑥)=0.6,
           then 𝑅(𝑥)=0.5⋅0.4+0.5⋅0.6=0.5
   -c. Threshold calibration: choose 𝜏_𝑅 on validation to maximize F1 in flagging true teacher-error cases.

5. Conformal Calibration (Orthogonal Approach)
   -a. Student residual:
       𝑟(𝑥_𝑖)=∣𝑆(𝑥_𝑖)−0.5∣
   -b. Miscoverage level 𝛽 (e.g. 𝛽=0.2 → allow 20% false alerts).
   -c. Threshold 𝜏_𝐶 : the (1−𝛽)-quantile of {𝑟(𝑥_𝑖)} on validation.
   -d. Flag rule: any new 𝑥 with ∣𝑆(𝑥)−0.5∣<𝜏_𝐶 is marked high-risk.
   -e. Guarantee: at most 𝛽% of non-errors will be incorrectly flagged.

6. Clustering Hard Cases for Insights
   -a. Select top 𝑝% of validation points by student residual 𝑟(𝑥) (e.g. 𝑝=10%).
   -b. Apply k-means with 𝑘=3 clusters to their feature vectors.
   -c. Compute cluster centers
       𝜇_𝑗 = 1/∣𝐶_𝑗∣ ∑_(𝑥_𝑖∈𝐶_𝑗) 𝑥_𝑖, 𝑗=1,2,3.
   -d. Interpretation: centers reveal prototypical profiles (e.g., “low product_rating & high discount_rate”),
                       highlighting why the model struggles in those segments.

7. Evaluation Metrics
   -a. Prediction uplift
       Δ_(acc)=Acc(𝑆)−Acc(𝑇)
       -1. Measures any net gain (or loss) in student accuracy versus teacher.
   -b. Detection performance (for each method: entropy-threshold, meta-gate, unified 𝑅, conformal)
       Precision=TP/(TP+FP), Recall=TP/(TP+FN), 𝐹1=2(Precision⋅Recall / Precision+Recall)
       -1. Tune each threshold to achieve the desired precision–recall trade-off (e.g., 80% precision & recall).

8. Summary of the Pipeline
   -a. Identify high-uncertainty points via teacher entropy.
   -b. Distill teacher → student with soft targets to retain accuracy in a lighter model.
   -c. Train a meta-model to gate known teacher failures.
   -d. Compute a unified risk score and calibrate a threshold to flag new high-risk cases.
   -e. Optionally, use conformal calibration for finite-sample error guarantees.
   -f. Cluster the hardest residuals to extract actionable patterns in feature space.
   -g. Evaluate both prediction uplift and detection metrics to validate effectiveness for production monitoring.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_recall_fscore_support
from scipy.stats import entropy

# Feature names reflecting business context
feature_names = [
    'order_amount', 'customer_tenure', 'num_transactions', 'credit_score',
    'days_since_last_purchase', 'product_rating', 'store_visits',
    'marketing_spend', 'discount_rate', 'seasonal_index'
]

csv_path = 'creditscore_data.csv'

# Load data from CSV
df = pd.read_csv(csv_path)

X = df[feature_names].values
y = df['target'].values

# Split into train/validation/test
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.4, random_state=1)
X_val, X_test, y_val, y_test   = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=1)

# -------------------------------------
# Teacher model + entropy uncertainty
# -------------------------------------
teacher = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
teacher.fit(X_train, y_train)
p_train_teacher = teacher.predict_proba(X_train)
p_val_teacher   = teacher.predict_proba(X_val)
p_test_teacher  = teacher.predict_proba(X_test)

pred_val_teacher  = (p_val_teacher[:,1] >= 0.5).astype(int)
pred_test_teacher = (p_test_teacher[:,1] >= 0.5).astype(int)

teacher_val_acc  = accuracy_score(y_val, pred_val_teacher)
teacher_val_ll   = log_loss(y_val, p_val_teacher[:,1])
teacher_test_acc = accuracy_score(y_test, pred_test_teacher)
teacher_test_ll  = log_loss(y_test, p_test_teacher[:,1])

actual_err_val  = pred_val_teacher != y_val
actual_err_test = pred_test_teacher != y_test

# -------------------------------------
# Student via distillation
# -------------------------------------
alpha = 0.7
Y_train_distill = alpha * y_train + (1 - alpha) * p_train_teacher[:,1]
student = MLPRegressor(hidden_layer_sizes=(32,16), max_iter=100, random_state=0)
student.fit(X_train, Y_train_distill)

p_val_student  = student.predict(X_val)
p_test_student = student.predict(X_test)
pred_val_student  = (p_val_student >= 0.5).astype(int)
pred_test_student = (p_test_student >= 0.5).astype(int)

student_val_acc  = accuracy_score(y_val, pred_val_student)
student_val_ll   = log_loss(y_val, p_val_student.clip(1e-6,1-1e-6))
student_test_acc = accuracy_score(y_test, pred_test_student)
student_test_ll  = log_loss(y_test, p_test_student.clip(1e-6,1-1e-6))

# -------------------------------------
# 4. Meta-model Gateway
# -------------------------------------
resid_train_teacher = (teacher.predict(X_train) != y_train).astype(int)
meta = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=1)
meta.fit(X_train, resid_train_teacher)
meta_proba_val  = meta.predict_proba(X_val)[:,1]
meta_proba_test = meta.predict_proba(X_test)[:,1]

# -------------------------------------
# Calibrate thresholds
# -------------------------------------
def calibrate_threshold(probs, actual, thr_values):
    best_f1, best_thr = -1, thr_values[0]
    for thr in thr_values:
        preds = probs >= thr
        _, _, f1, _ = precision_recall_fscore_support(actual, preds, average='binary')
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr

ent_val  = entropy(p_val_teacher.T) / np.log(2)
ent_test = entropy(p_test_teacher.T) / np.log(2)
thr_ent  = np.quantile(ent_val, 0.9)
thr_meta = calibrate_threshold(meta_proba_val, actual_err_val, np.linspace(0,1,101))

risk_val  = 0.5 * ent_val + 0.5 * meta_proba_val
risk_test = 0.5 * ent_test + 0.5 * meta_proba_test
thr_risk  = calibrate_threshold(risk_val, actual_err_val, np.linspace(risk_val.min(), risk_val.max(), 101))

resid_val_student  = np.abs(p_val_student - y_val)
resid_test_student = np.abs(p_test_student - y_test)
thr_conf_90 = np.quantile(resid_val_student, 0.9)
thr_conf_80 = np.quantile(resid_val_student, 0.8)

# -------------------------------------
# Apply flags for detection
# -------------------------------------
flag_ent_val    = ent_val >= thr_ent
flag_ent_test   = ent_test >= thr_ent
flag_meta_val   = meta_proba_val >= thr_meta
flag_meta_test  = meta_proba_test >= thr_meta
flag_risk_val   = risk_val >= thr_risk
flag_risk_test  = risk_test >= thr_risk
flag_conf_90_val = resid_val_student >= thr_conf_90
flag_conf_80_val = resid_val_student >= thr_conf_80
flag_conf_90_test = resid_test_student >= thr_conf_90
flag_conf_80_test = resid_test_student >= thr_conf_80

# Detection metrics
def detect_metrics(flag, actual):
    tp = np.sum(flag & actual)
    fp = np.sum(flag & ~actual)
    fn = np.sum(~flag & actual)
    return tp/(tp+fp) if tp+fp>0 else 0, tp/(tp+fn) if tp+fn>0 else 0

# -------------------------------------
# Summarize results
# -------------------------------------
perf = pd.DataFrame({
    "Model":   ["Teacher","Student","Teacher","Student"],
    "Dataset": ["Val","Val","Test","Test"],
    "Accuracy":[teacher_val_acc, student_val_acc, teacher_test_acc, student_test_acc],
    "LogLoss": [teacher_val_ll, student_val_ll, teacher_test_ll, student_test_ll]
})

methods = {
    "Entropy":   (flag_ent_val, actual_err_val, flag_ent_test, actual_err_test),
    "Meta-Gate": (flag_meta_val, actual_err_val, flag_meta_test, actual_err_test),
    "Unified":   (flag_risk_val, actual_err_val, flag_risk_test, actual_err_test),
    "Conf_90%":  (flag_conf_90_val, actual_err_val, flag_conf_90_test, actual_err_test),
    "Conf_80%":  (flag_conf_80_val, actual_err_val, flag_conf_80_test, actual_err_test)
}

detect_records = []
for name, (fv, av, ft, at) in methods.items():
    pv, rv = detect_metrics(fv, av)
    pt, rt = detect_metrics(ft, at)
    detect_records.append({"Method":name,"Dataset":"Val","Precision":pv,"Recall":rv})
    detect_records.append({"Method":name,"Dataset":"Test","Precision":pt,"Recall":rt})
detect_df = pd.DataFrame(detect_records)

# Feature mean comparison & clustering
df_val = pd.DataFrame(X_val, columns=feature_names)
df_val["err"] = resid_val_student
high_df = df_val.nlargest(int(0.1*len(df_val)), "err")
comparison = pd.DataFrame({
    "overall_mean": df_val.mean(), 
    "high_err_mean": high_df.mean()
})
kmeans = KMeans(n_clusters=3, random_state=0).fit(high_df.drop(columns="err"))
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=feature_names)

# -------------------------------------
# Output and Recommended Next Actions
# -------------------------------------
print("\n--- Prediction Performance ---")
print(perf)

print("\n--- High-Error Detection ---")
print(detect_df)

print("\n--- Feature Mean Comparison (Val) ---")
print(comparison)

print("\n--- Next Steps: Conformal Thresholds ---")
print(pd.DataFrame([
    {"Threshold":"Conf_90%","Precision":detect_metrics(flag_conf_90_val, actual_err_val)[0],
     "Recall":detect_metrics(flag_conf_90_val, actual_err_val)[1]},
    {"Threshold":"Conf_80%","Precision":detect_metrics(flag_conf_80_val, actual_err_val)[0],
     "Recall":detect_metrics(flag_conf_80_val, actual_err_val)[1]}
]))

print("\n--- Next Steps: Unified Score Weights ---")
for w in [0.3, 0.5, 0.7]:
    risk = w*ent_val + (1-w)*meta_proba_val
    thr = calibrate_threshold(risk, actual_err_val, np.linspace(risk.min(), risk.max(), 101))
    p_w, r_w = detect_metrics(risk >= thr, actual_err_val)
    print(f"Weight_ent={w}: Precision={p_w:.3f}, Recall={r_w:.3f}")

print("\n--- Next Steps: High-Error Clusters (centers) ---")
print(cluster_centers)

print("\n--- Recommended Next Actions ---")
print("1. Choose the Conformal 80% threshold for balanced precision/recall in production.")
print("2. Monitor flagged observations over time to validate detection rates out-of-sample.")
print("3. Engineer targeted features or specialist models for the identified high-error clusters.")
print("4. Automate periodic recalibration of conformal thresholds as data distributions drift.")
