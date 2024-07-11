"""
From https://towardsdatascience.com/doping-a-technique-to-test-outlier-detectors-3f6b847ab8d4

1. Challenges of Outlier Detection
   - Evaluating outlier detection is difficult because it involves working with unlabeled data.
   - Unlike predictive models (regression, classification), outlier detection lacks clear evaluation metrics.

2. Comparison with Clustering
   - Clustering, like outlier detection, deals with unlabeled data but can measure cluster quality using metrics such as the Silhouette score.
   - These metrics assess the consistency within clusters and the differences between clusters.

3. Limitations in Evaluating Outlier Detection
   - Any method to evaluate outliers tends to be subjective, and no definitive definition of outliers exists.
   - Measures like entropy can be used but are inherently part of outlier detection algorithms themselves, making evaluation circular.

4. Creating Doped Data
   - Doping involves modifying existing data records to create outliers.
   - For example, changing a single cell to an unusual value or creating unusual combinations of values across multiple cells.
   - Doped data helps test the effectiveness of different outlier detectors.

5. Using Doped Data
   - Doped records can be included in training data or used exclusively in test data.
   - When included in training data, the goal is to see if detectors can identify these records as outliers.
   - When used only in testing, it helps evaluate the detector's ability to find outliers in new data.

6. Algorithms to Create Doped Data
   - One method is to randomly select and modify cell values.
   - While some doped records may not be true outliers, most will disrupt the normal associations between features, 
     making them effective for testing.

7. Alternative Doping Methods
   - Categorical Data
     Select a new value that differs from both the original value and the predicted value based on other row features.
     Use a predictive model (e.g., Random Forest Classifier) to predict the current value and ensure the new value is distinct.
   - Numerical Data:
     Divide numeric features into quartiles (or quantiles, at least three).
     Select a new value in a different quartile than both the original and the predicted values.
     For example, if the original value is in Q1 and the predicted value is in Q2, select a value in Q3 or Q4 to ensure
     it goes against the normal feature relationships.
     
8. Creating a Suite of Test Datasets
    - Doped records can vary in the number and extent of modifications to simulate different levels of anomaly.
    - Create multiple test suites with varying levels of difficulty to evaluate detectors more accurately.
      1. Obvious Anomalies: Multiple features modified significantly from their original values.
      2. Subtle Anomalies: A single feature modified slightly from its original value.
   - Different test sets can target different features, as some may be more relevant or easier to detect anomalies in.

9. Purpose and Use of Test Suites
   - Ensure that doping reflects the types of outliers of interest in real data.
   - Cover the range of anomalies you are interested in detecting.
   - Multiple test sets help in selecting the best-performing detectors and estimating their future performance.
   - Understand that the number of detected outliers, false positives, and false negatives depend heavily on the actual data, 
     which is unpredictable.

10. Creating Effective Ensembles of Outlier Detectors
    - Ensembles are usually necessary for reliable outlier detection as different detectors catch different types of outliers.
    - Understanding which types of outliers each detector can detect helps in forming an effective ensemble.
   - Avoid redundancy and ensure comprehensive detection by combining detectors with complementary strengths.

Conclusion
The article elaborates on doping methods to create robust test datasets that simulate various types of anomalies.
By using these datasets, outlier detectors can be evaluated more accurately, aiding in the selection and improvement
of outlier detection systems. Additionally, understanding each detector's strengths helps in creating effective ensembles,
which are crucial for thorough and reliable outlier detection in diverse datasets.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import seaborn as sns
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ecod import ECOD

# Collect the data
data = fetch_openml('abalone', version=1) 
df = pd.DataFrame(data.data, columns=data.feature_names)
df = pd.get_dummies(df)
df = pd.DataFrame(RobustScaler().fit_transform(df), columns=df.columns)

# Use an Isolation Forest to clean the data
clf = IForest() 
clf.fit(df)
if_scores = clf.decision_scores_
top_if_scores = np.argsort(if_scores)[::-1][:10]
clean_df = df.loc[[x for x in df.index if x not in top_if_scores]].copy()

# Create a set of doped records
doped_df = df.copy() 
for i in doped_df.index:
  col_name = np.random.choice(df.columns)
  med_val = clean_df[col_name].median()
  if doped_df.loc[i, col_name] > med_val:
    doped_df.loc[i, col_name] = \   
      clean_df[col_name].quantile(np.random.random()/2)
  else:
    doped_df.loc[i, col_name] = \
       clean_df[col_name].quantile(0.5 + np.random.random()/2)

# Define a method to test a specified detector. 
def test_detector(clf, title, df, clean_df, doped_df, ax): 
  clf.fit(clean_df)
  df = df.copy()
  doped_df = doped_df.copy()
  df['Scores'] = clf.decision_function(df)
  df['Source'] = 'Real'
  doped_df['Scores'] = clf.decision_function(doped_df)
  doped_df['Source'] = 'Doped'
  test_df = pd.concat([df, doped_df])
  sns.boxplot(data=test_df, orient='h', x='Scores', y='Source', ax=ax)
  ax.set_title(title)

# Plot each detector in terms of how well they score doped records 
# higher than the original records
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(10, 3)) 
test_detector(IForest(), "IForest", df, clean_df, doped_df, ax[0])
test_detector(LOF(), "LOF", df, clean_df, doped_df, ax[1])
test_detector(ECOD(), "ECOD", df, clean_df, doped_df, ax[2])
plt.tight_layout()
plt.show()
