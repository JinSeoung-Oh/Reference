### From https://medium.com/data-science-collective/attention-based-neural-network-distillation-enhancing-performance-through-learned-weighted-c4efa612b257

"""
1. Introduction
   -a. Problem: Traditional knowledge distillation uses a single “teacher” model, 
                losing out on complementary strengths and uncertainties of other models.
   -b. Solution: An attention-based distillation approach that dynamically aggregates predictions 
                 from multiple teacher models (e.g. XGBoost and Random Forest) so that a student neural network benefits
                 from each teacher’s specialties in different contexts.

1 — Why Attention?
    -a. Standard distillation treats multiple teachers equally or with fixed weights—too rigid for real‑world heterogeneity.
    -b. Attention lets the student learn input‑dependent weights over teachers:
        -1. E.g. one teacher excels on holiday data, another on weekdays—attention shifts based on features.
        -2. Results in more robust, context‑aware predictions.

2 — Framework & Mechanisms
    2.1 Attention‐Based Neural Distillation
        -a. Teachers’ predictions: 𝑌_𝑗(𝑋) from each teacher 𝑗
        -b. Shared embedding: ℎ=𝐺(𝑋) via a neural feature extractor.
        -c. Attention weights 𝑎_𝑗(𝑋):
            𝑎_𝑗(𝑋)= (exp⁡(𝐹_𝑗(ℎ)) / ∑_𝑘 exp⁡(𝐹_𝑘(ℎ)),
            where each 𝐹_𝑗 is a small network estimating teacher 𝑗’s reliability for input 𝑋
        -d. Teacher aggregation:
             𝑌_(mix)(𝑋)=∑_𝑗 𝑎_𝑗(𝑋)𝑌_𝑗(𝑋)
    2.2 Student Model & Losses
        -a. Student outputs two streams from ℎ
            -1. Direct prediction 𝑠_(out)=𝜎(𝑊_𝑠 ℎ)
            -2. Aggregated prediction 𝑡_(out)=𝑌_(mix)(𝑋)
        -b. Final output
            𝑦^(𝑋)=1/2(𝑠_(out)+𝑡_(out))
        -c. Loss =
            𝐿=BCE(𝑦^,𝑦)+𝜆KL(𝑌_(mix)∥𝑠_(out))
        where BCE is Binary Cross‑Entropy and KL is Kullback–Leibler divergence, encouraging both accuracy 
        and fidelity to the teachers’ distribution.

3 — Code Experiment
    A synthetic e‑commerce dataset with features:
    Age, Income, Days since last interaction, Holiday flag, Loyalty score, Marketing channel, Promo flag, Purchase label
"""
    # 3.1 Data Preparation & Teachers
      import numpy as np, pandas as pd
      from sklearn.preprocessing import OneHotEncoder, StandardScaler
      from xgboost import XGBClassifier
      from sklearn.ensemble import RandomForestClassifier
      from sklearn.model_selection import train_test_split

      # 1. Load & split promo vs non‑promo
      df = pd.read_csv('sales.csv')
      train_df = df[df.Promo == 0]
      test_df  = df[df.Promo == 1]

      # 2. One‑hot encode Channel
      ohe      = OneHotEncoder()
      ch_tr    = ohe.fit_transform(train_df[['Channel']]).toarray()
      ch_te    = ohe.transform(test_df[['Channel']]).toarray()

      # 3. Assemble feature matrices & labels
      X_train = pd.concat([train_df[['Age','Income','Days','Holiday','Loyalty']].reset_index(drop=True),
                           pd.DataFrame(ch_tr).reset_index(drop=True)], axis=1)
      y_train = train_df['Purchase'].values
      X_test  = pd.concat([test_df [['Age','Income','Days','Holiday','Loyalty']].reset_index(drop=True),
                           pd.DataFrame(ch_te).reset_index(drop=True)], axis=1)
      y_test  = test_df['Purchase'].values

      # 4. Scale and split
      scaler = StandardScaler()
      X_train = scaler.fit_transform(X_train)
      X_test  = scaler.transform(X_test)
      X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

      # 5. Train teachers
      xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
      rf  = RandomForestClassifier()
      xgb.fit(X_tr, y_tr)
      rf.fit (X_tr, y_tr)

      # 6. Collect teacher prob‑predictions
      def get_teacher_preds(X):
          return np.vstack([
              xgb.predict_proba(X)[:,1],
              rf.predict_proba(X)[:,1]
          ]).T

      tp_tr   = get_teacher_preds(X_tr)
      tp_val  = get_teacher_preds(X_val)
      tp_test = get_teacher_preds(X_test)

# 3.2 Bayesian Distillation Student
      import torch, torch.nn as nn, torch.optim as optim
      from sklearn.metrics import roc_auc_score, accuracy_score
      from scipy.stats import ks_2samp

      class BayesianDistillNN(nn.Module):
          def __init__(self, input_dim, n_teachers):
              super().__init__()
              self.shared  = nn.Sequential(
                  nn.Linear(input_dim, 64), nn.ReLU(),
                  nn.Linear(64, 32), nn.ReLU()
              )
              self.attn    = nn.Sequential(nn.Linear(32, n_teachers), nn.Softmax(dim=1))
              self.student = nn.Sequential(nn.Linear(32, 1), nn.Sigmoid())

          def forward(self, x, teacher_probs):
              h    = self.shared(x)
              a    = self.attn(h)                               # (batch, n_teachers)
              t_out= torch.sum(a * teacher_probs, dim=1, keepdim=True)
              s_out= self.student(h)
              return 0.5*(s_out + t_out), a

      #### Training Loop
      # Tensor conversion
      X_tr_t   = torch.tensor(X_tr,   dtype=torch.float32)
      y_tr_t   = torch.tensor(y_tr,   dtype=torch.float32).unsqueeze(1)
      tp_tr_t  = torch.tensor(tp_tr,  dtype=torch.float32)
      X_val_t  = torch.tensor(X_val,  dtype=torch.float32)
      y_val_t  = torch.tensor(y_val,  dtype=torch.float32).unsqueeze(1)
      tp_val_t = torch.tensor(tp_val, dtype=torch.float32)

      model    = BayesianDistillNN(X_tr.shape[1], 2)
      opt      = optim.Adam(model.parameters(), lr=5e-4)
      bce      = nn.BCELoss()

      best_auc, patience = 0.0, 10
      for epoch in range(50):
          model.train()
          opt.zero_grad()
          pred, att = model(X_tr_t, tp_tr_t)
          teacher_mix = torch.sum(att * tp_tr_t, dim=1, keepdim=True)
          kl_loss = torch.mean(teacher_mix * torch.log((teacher_mix+1e-7)/(pred+1e-7)))
          loss    = bce(pred, y_tr_t) + 0.3*kl_loss
          loss.backward(); opt.step()

          # Validation
          with torch.no_grad():
              model.eval()
              val_pred, _ = model(X_val_t, tp_val_t)
              val_auc = roc_auc_score(y_val, val_pred.numpy())
              if val_auc > best_auc:
                  best_auc = val_auc
                  torch.save(model.state_dict(), 'best_bayes.pt')
                  patience = 10
              else:
                  patience -= 1
                  if patience == 0: break
          print(f"Epoch {epoch+1}, Loss {loss:.4f}, Val AUC {val_auc:.4f}")

## 3.3 Evaluation
      # Load best model
      model.load_state_dict(torch.load('best_bayes.pt'))
      model.eval()
      with torch.no_grad():
          test_pred, att = model(torch.tensor(X_test, dtype=torch.float32),
                                 torch.tensor(tp_test, dtype=torch.float32))
          pred_np = test_pred.numpy().ravel()
          auc     = roc_auc_score(y_test, pred_np)
          acc     = accuracy_score(y_test, pred_np>0.5)
          ks      = ks_2samp(pred_np[y_test==1], pred_np[y_test==0])[0]
          att_mean= att.numpy().mean(axis=0)

      print(f"AUC: {auc:.4f}, Accuracy: {acc:.4f}, KS: {ks:.4f}")
      print(f"Avg attention (XGB, RF): {att_mean}")
      # Teachers alone
      for name, model_t in {'XGB':xgb,'RF':rf}.items():
          p = model_t.predict_proba(X_test)[:,1]
          print(f"{name} — AUC {roc_auc_score(y_test,p):.4f}, "
                f"Acc {accuracy_score(y_test,p>0.5):.4f}, "
                f"KS {ks_2samp(p[y_test==1],p[y_test==0])[0]:.4f}")

"""
4 — Results & Insights
    Model |	AUC |	Accuracy	| KS Score
    Bayesian Distillation |	0.7928 |	0.8996	| 0.4505
    XGBoost	| 0.7889 |	0.5143	| 0.4352
    Random Forest	| 0.7691	| 0.5077 |	0.4079
    
    -a. Distilled student outperforms both teachers on all metrics.
    -b. Attention weights ≈ [0.483, 0.517] for XGB vs RF—demonstrating a learned, near‑equal blend.
    -c. Teachers alone had decent ranking (AUC) but poor thresholded accuracy (~51%).

5 — Take‑Home & Future Directions
    -a. Dynamic aggregation via attention outperforms fixed or equal‐weight ensembling.
    -b. Compact student: captures teachers’ strengths in a single, efficient model suitable for real‑time use.
    -c. Applications: customer analytics, fraud detection, credit scoring, personalized marketing.
    -d. Extensions: integrate uncertainty modeling (Bayesian methods), domain adaptation, online continual distillation.

