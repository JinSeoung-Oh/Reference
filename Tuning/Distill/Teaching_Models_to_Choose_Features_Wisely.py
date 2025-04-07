### From https://medium.com/data-science-collective/teaching-models-to-choose-features-wisely-a-distill-to-select-approach-a9359e2ba5d1

import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.stats import ks_2samp
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

###########################
# Load Data
###########################

df = pd.read_csv('purchase_binary.csv')

# Use only treatment data
df = df[df["Promo"] == 1]  

###########################
# Data Preprocessing
###########################
df_encoded = pd.get_dummies(df, columns=["Channel"], drop_first=True)
print("Dummy columns:", [col for col in df_encoded.columns if "Channel_" in col])
# For instance, assume columns "Channel_Mobile" and "Channel_Online" exist.
features = ["Age", "Income", "Days", "Holiday", "Loyalty", "Channel_Mobile", "Channel_Online"]
target = "Purchase"

X = df_encoded[features].values.astype(np.float32)
y = df_encoded[target].values.astype(np.int64)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

###########################
# Train-Test Split (65% / 35%)
###########################
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.35, random_state=42, stratify=y)

###########################
# Train the Teacher Model (LightGBM)
###########################
teacher = lgb.LGBMClassifier(random_state=42)
teacher.fit(X_train, y_train)

teacher_train_prob = teacher.predict_proba(X_train)[:, 1]
teacher_test_prob = teacher.predict_proba(X_test)[:, 1]

teacher_auc = roc_auc_score(y_test, teacher_test_prob)
teacher_acc = accuracy_score(y_test, teacher.predict(X_test))
teacher_ks = ks_2samp(teacher_test_prob[y_test == 1], teacher_test_prob[y_test == 0]).statistic

print("\nTeacher (LightGBM) Performance:")
print(f"AUC = {teacher_auc:.4f}, KS = {teacher_ks:.4f}, ACC = {teacher_acc:.4f}")

###########################
# Define the Student Model (Logistic Regression)
###########################
class LogisticRegressionStudent(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionStudent, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        logit = self.linear(x)
        prob = torch.sigmoid(logit)
        return prob

student = LogisticRegressionStudent(input_dim=X_train.shape[1])

###########################
# Define the Combined Loss Function
###########################
def kl_divergence(p_teacher, p_student, eps=1e-8):
    p_teacher = torch.clamp(p_teacher, eps, 1.0 - eps)
    p_student = torch.clamp(p_student, eps, 1.0 - eps)
    kl = p_teacher * torch.log(p_teacher / p_student) + (1 - p_teacher) * torch.log((1 - p_teacher) / (1 - p_student))
    return torch.mean(kl)

# Hyperparameters for distillation
gamma = 1.0       # weight for KL loss
lambda_l1 = 1e-4  # weight for L1 regularization

###########################
# Train the Student Model (Distillation) with Early Stopping
###########################
X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train.reshape(-1, 1)).float()
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

optimizer = optim.Adam(student.parameters(), lr=0.001)
num_epochs = 100

# For early stopping
best_val_loss = float('inf')
patience = 10
best_epoch = 0
best_student_state = None

# Create validation tensors from X_test and y_test
X_val_tensor = torch.from_numpy(X_test)
y_val_tensor = torch.from_numpy(y_test.reshape(-1, 1)).float()

student.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        student_prob = student(batch_X)  # Student predictions
        
        # For simplicity, we use the first N teacher_train_prob values for this batch.
        batch_indices = np.arange(batch_X.shape[0])
        teacher_probs = torch.from_numpy(teacher_train_prob[batch_indices]).float().unsqueeze(1)
        
        # Compute BCE loss (true labels)
        bce_loss = nn.BCELoss()(student_prob, batch_y)
        # Compute KL divergence loss (distillation)
        kl_loss = kl_divergence(teacher_probs.squeeze(), student_prob.squeeze())
        # Compute L1 penalty on the student's weights (LASSO)
        l1_loss = torch.norm(student.linear.weight, 1)
        
        loss = bce_loss + gamma * kl_loss + lambda_l1 * l1_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_X.size(0)
    epoch_loss /= len(train_dataset)
    
    # Evaluate validation loss on the test set
    student.eval()
    with torch.no_grad():
        val_prob = student(X_val_tensor)
        val_loss = nn.BCELoss()(val_prob, y_val_tensor).item()
    student.train()
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {epoch_loss:.4f} - Val Loss: {val_loss:.4f}")
    
    # Check early stopping condition
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_student_state = student.state_dict()
    elif epoch - best_epoch >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break

# Load best model state
if best_student_state is not None:
    student.load_state_dict(best_student_state)

###########################
# Evaluate the Student Model on Test Data
###########################
student.eval()
with torch.no_grad():
    X_test_tensor = torch.from_numpy(X_test)
    student_test_prob = student(X_test_tensor).cpu().numpy().flatten()

student_pred = (student_test_prob >= 0.5).astype(int)
student_auc = roc_auc_score(y_test, student_test_prob)
student_acc = accuracy_score(y_test, student_pred)
student_ks = ks_2samp(student_test_prob[y_test == 1], student_test_prob[y_test == 0]).statistic

print("\nStudent (Distilled Logistic Regression) Performance:")
print(f"AUC = {student_auc:.4f}, KS = {student_ks:.4f}, ACC = {student_acc:.4f}")

###########################
# Model Interpretability: Print Coefficients and Feature Importance
###########################
# Extract weight and bias from the student's linear layer
weights = student.linear.weight.detach().cpu().numpy().flatten()
bias = student.linear.bias.detach().cpu().numpy()[0]

# Pair each feature with its coefficient and sort by absolute value
feature_importance = list(zip(features, weights))
feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)

print("\nStudent Model Coefficients and Feature Importance:")
print(f"Intercept (bias): {bias:.4f}")
for feature, coef in feature_importance:
    print(f"Feature: {feature:20s} Coefficient: {coef:.4f}")

# Define a threshold for feature selection (e.g., absolute coefficient > 0.1)
threshold = 0.1
selected_features = [feat for feat, coef in feature_importance if abs(coef) > threshold]
print("\nFeatures Selected (|coefficient| > 0.1):")
print(selected_features)


