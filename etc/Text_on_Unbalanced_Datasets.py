### From https://generativeai.pub/comprehensive-guide-to-handling-unbalanced-datasets-2e01abd5354e
"""
1. Introduction to Unbalanced Datasets
1.1 What is an Unbalanced Dataset? An unbalanced dataset refers to classification problems where classes are not equally represented. 
    Typically, one class (majority class) vastly outnumbers the other (minority class). Examples include:

    - Fraud detection: 99.9% legitimate vs. 0.1% fraudulent transactions.
    - Medical diagnosis: healthy patients vs. patients with rare conditions.
    - Spam detection: non-spam vs. spam emails.
1.2 Why are Unbalanced Datasets Problematic? Several challenges arise:
    - Bias towards the majority class: Models may achieve high accuracy by predicting only the majority class, which is unhelpful.
    - Poor performance on the minority class: Algorithms may fail to capture the minority class's characteristics.
    - Misleading evaluation metrics: Metrics like accuracy can be misleading, as high accuracy may hide poor performance on the minority class.
    - Overfitting risk: Models may overfit to the majority class, failing to generalize well to new, minority-class data.

2. Techniques for Handling Unbalanced Datasets
2.1 Data-Level Methods: Resampling
  2.1.1 Oversampling the Minority Class
  - Random Oversampling
    Duplicates minority class examples. 
    - Pros: simple to implement. 
    - Cons: may lead to overfitting.
  - SMOTE (Synthetic Minority Over-sampling Technique)
    Generates synthetic minority samples by interpolating between minority class neighbors. 
    - Pros: reduces overfitting risk. 
    - Cons: may increase noise.
  - ADASYN (Adaptive Synthetic)
    Similar to SMOTE but focuses on generating synthetic samples near the decision boundary.
    - Pros: adapts to data, improving learning on hard-to-classify instances. 
    - Cons: computationally expensive.

  2.1.2 Undersampling the Majority Class
  - Random Undersampling
    Removes random samples from the majority class. 
    - Pros: reduces training time. 
    - Cons: may lose important information.
  - Tomek Links
    Removes majority class examples that are close to minority ones, helping clean class boundaries. 
    - Cons: may have limited effect.
  - Cluster Centroids
    Uses K-means clustering to reduce majority class data, replacing clusters with their centroids. 
    - Pros: retains more information. 
    - Cons: computationally expensive for large datasets.

  2.1.3 Combination Methods
  - SMOTEENN: Combines SMOTE with Edited Nearest Neighbors (ENN) to clean up noisy data after oversampling.
  - SMOTETomek: Combines SMOTE with Tomek Links, improving both class balancing and boundary cleaning.

2.2 Algorithm-Level Approaches
  2.2.1 Adjust Class Weights
  - Assign higher weights to the minority class during training. Most models, such as logistic regression, support this feature.
    - Pros: easy to implement and doesn't modify data. 
    - Cons: sensitive to the choice of weights.

  2.2.2 Modify the Algorithm
  - Cost-Sensitive Learning: Penalizes misclassification of the minority class more heavily.
  - One-Class Learning: Trains on the majority class and treats the minority class as anomalies.
  - Threshold Moving: Adjusts the classification threshold to favor the minority class.

2.3 Ensemble Methods
  2.3.1 Balanced Random Forest
  Uses random undersampling in each bootstrap sample to balance the training data for each decision tree in the random forest. 
  - Pros: combines random forest strengths with balanced data. 
  - Cons: may not use all majority class examples.

  2.3.2 Easy Ensemble
  Creates multiple balanced datasets by undersampling the majority class and trains a separate model on each subset. 
  - Pros: uses all available data while balancing. 
  - Cons: computationally expensive.

2.4 Advanced Techniques
  2.4.1 Generative Adversarial Networks (GANs)
  Uses GANs to generate synthetic minority class examples. 
  - Pros: can create high-quality synthetic samples.
  - Cons: difficult to implement and train.

  2.4.2 Two-Phase Learning
  - Phase 1: Train on balanced data to learn general patterns.
  - Phase 2: Fine-tune on the original unbalanced data. Pros: combines benefits of balanced and unbalanced training.
    - Cons: requires careful balancing between phases.

3. Evaluation Metrics for Unbalanced Datasets
   3.1 Confusion Matrix
   Shows True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).

   3.2 Precision, Recall, and F1-Score
   Precision: TP / (TP + FP).
   Recall: TP / (TP + FN).
   F1-Score: Harmonic mean of precision and recall.

   3.3 ROC AUC
   Measures a model’s ability to distinguish between classes across various thresholds.

   3.4 Precision-Recall Curve
   Particularly useful for unbalanced datasets as ROC curves can be overly optimistic.

4. Best Practices for Handling Unbalanced Datasets
   - Understand Your Data: Analyze the imbalance's extent.
   - Choose Appropriate Techniques: Select methods based on imbalance severity and domain knowledge.
   - Cross-Validation: Use stratified k-fold cross-validation to maintain class proportions across folds.
   - Combine Methods: Combining approaches often yields the best results.
   - Monitor Real-world Performance: Ensure the model generalizes well to new data.
   - Consider Domain Knowledge: Imbalance may reflect the real-world scenario and shouldn’t always be altered.
   - Experiment and Iterate: Try different techniques and compare performance.

5. Case Study: Credit Card Fraud Detection
   - Problem Description: Detecting fraudulent transactions in a highly unbalanced dataset (0.1% fraud).
   - Approach:
     Data Exploration: Analyzed class distribution (99.9% legitimate vs. 0.1% fraud).
     Preprocessing: Used mean imputation for missing values, scaled features using StandardScaler.
     Resampling: Applied SMOTE to increase fraud cases from 0.1% to 10%.
     Model Selection: Used a Random Forest model with adjusted class weights, assigning higher weight to fraud cases.
     Evaluation: Used stratified 5-fold cross-validation and focused on precision-recall and F1-score as the main metrics.
"""
       
### Example 
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, precision_recall_curve

# Assume X and y are your features and target
smote = SMOTE(sampling_strategy=0.1)  # Increase minority class to 10%
X_resampled, y_resampled = smote.fit_resample(X, y)
model = RandomForestClassifier(class_weight={0: 1, 1: 10})
# Cross-validation
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5, scoring='f1')
# Train final model
model.fit(X_resampled, y_resampled)
# Predictions on test set
y_pred = model.predict(X_test)
# Evaluate
f1 = f1_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
# Plot precision-recall curve
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()


