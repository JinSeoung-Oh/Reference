### From https://medium.com/chat-gpt-now-writes-all-my-articles/the-less-known-machine-learning-tree-yggdrasil-gradient-boosted-trees-learner-830713e5d5a6

"""
1. What Are Gradient Boosted Trees (GBTs)? 
   Gradient Boosted Trees are an ensemble learning technique that trains shallow decision trees sequentially. 
   Each new tree focuses on the errors (gradients) made by the existing ensemble, refining predictions step-by-step. 
   This iterative process allows GBTs to minimize a chosen loss function effectively. 
   GBTs are widely used for classification, regression, and ranking tasks due to their flexibility and strong predictive performance.

2. Yggdrasil Decision Forests (YDF): YDF is a library that streamlines the training and deployment of decision forest models. It:
   -a. Works with Python, C++, and TensorFlow.
   -b. Supports various ML tasks like classification, regression, ranking, and uplift modeling.
   -c. Can evaluate, interpret, and serve decision forest models easily.
   -d. Handles diverse data types (numerical, categorical, text) and missing values with minimal preprocessing.
   -e. Integrates seamlessly with TensorFlow Decision Forests, enhancing accessibility for users already familiar with TensorFlow.
   -f. Interoperates smoothly with popular Python data libraries like Pandas.

3. Getting Started with GBTs Using YDF: YDF simplifies implementing Gradient Boosted Trees. Its user-friendly Python API lets data scientists and engineers 
   quickly build and refine GBT models, taking advantage of automatic handling of mixed data types and the option to serve models in production environments.
"""

! pip install ydf -U

import ydf  # Yggdrasil Decision Forests
import pandas as pd  # For data handling

ds_path = "https://raw.githubusercontent.com/google/yggdrasil-decision-forests/main/yggdrasil_decision_forests/test_data/dataset"
train_ds = pd.read_csv(f"{ds_path}/adult_train.csv")
test_ds = pd.read_csv(f"{ds_path}/adult_test.csv")

# Preview the dataset
train_ds.head()

### Training the Model
model = ydf.GradientBoostedTreesLearner(label="income").train(train_ds)

### Making Predictions
individual_prediction = model.predict({
    'age': [39],
    'workclass': ['State-gov'],
    'fnlwgt': [77516],
    'education': ['Bachelors'],
    'education_num': [13],
    'marital_status': ['Never-married'],
    'occupation': ['Adm-clerical'],
    'relationship': ['Not-in-family'],
    'race': ['White'],
    'sex': ['Male'],
    'capital_gain': [2174],
    'capital_loss': [0],
    'hours_per_week': [40],
    'native_country': ['United-States'],
})
print(individual_prediction)

### Model Evaluation
evaluation = model.evaluate(test_ds)
print(f"Test Accuracy: {evaluation.accuracy}")
print("Full Evaluation Report:")
print(evaluation)
