"""
From https://towardsdatascience.com/reinforcement-learning-for-feature-selection-be1e7eeb0acc

Feature selection is a critical step in building a machine learning model. 
Choosing the right features can significantly improve model performance, 
especially when dealing with high-dimensional datasets. 

This article introduces a novel method for feature selection using reinforcement learning (RL) and the Markov Decision Process (MDP),
implemented in the Python library FSRLearning.

1. Reinforcement Learning and Markov Decision Process for Feature Selection
   Reinforcement learning, particularly through the lens of MDP, is an innovative approach in data science for feature selection. 
   Unlike traditional methods, RL-based feature selection involves an agent interacting with an environment, making decisions to maximize a reward.
   Here’s how it translates to feature selection:

  -1. State: A state represents a subset of features from the dataset. 
      For example, if the dataset has features Age, Gender, and Height, possible states include:

      [] (empty set)
      [Age], [Gender], [Height] (1-feature sets)
      [Age, Gender], [Gender, Height], [Age, Height] (2-feature sets)
      [Age, Gender, Height] (all features)

  -2. Action: An action involves adding a new feature to the current subset. For instance:
      From [Age] to [Age, Gender]
      From [Gender, Height] to [Age, Gender, Height]

  -3. Reward: The reward measures the quality of a state, such as the increase in model accuracy when a new feature is added. For example:
      Accuracy with [Age] = 0.65
      Accuracy with [Age, Gender] = 0.76
      Reward for adding Gender = 0.76 - 0.65 = 0.11

2. Implementation and Python Library FSRLearning
   The FSRLearning library facilitates the RL-based feature selection process. Here’s a simplified implementation and usage guide:

   1. Installation:
      pip install fsrlearning

   2. Defining the Environment:
      Define your states, actions, and rewards using the library’s API.

   3. Training the Model:
      Implement an RL agent that explores different feature subsets and updates based on received rewards.

   4. Epsilon-Greedy Algorithm:
      The algorithm chooses the next state either randomly (with a probability epsilon) or by selecting the action that maximizes accuracy.

      epsilon = 0.2  # Probability for random action selection
      if np.random.rand() < epsilon:
          next_state = random.choice(possible_next_states)
      else:
           next_state = max(possible_next_states, key=lambda x: average_reward[x])

    5. Updating Rewards:
       Maintain and update a list of average rewards for each feature.

       average_reward[feature] = (average_reward[feature] * k + reward) / (k + 1)

    6. Stopping Conditions:
       Stop when all features are selected.
       Stop if a sequence of visited states shows degrading performance.

3. Practical Example
   Consider a dataset with features Age, Gender, and Height. The RL agent starts with an empty set and evaluates different subsets:

from fsrlearning import FeatureSelectionEnv

# Initialize environment
env = FeatureSelectionEnv(data, target)

# Define initial state and parameters
state = []
epsilon = 0.2

# Run the RL-based feature selection process
while not env.is_terminal(state):
    if np.random.rand() < epsilon:
        next_state = env.random_action(state)
    else:
        next_state = env.greedy_action(state)

    reward = env.evaluate(next_state)
    env.update_average_reward(next_state, reward)
    state = next_state

# Retrieve the selected features
selected_features = env.get_selected_features()
print("Selected features:", selected_features)

Conclusion
Reinforcement learning offers a powerful and efficient method for feature selection, especially in high-dimensional datasets.
By treating feature selection as an MDP, we can systematically explore and identify the most meaningful features, 
optimizing model performance while minimizing computational costs.
The FSRLearning library provides the necessary tools to implement this approach, making advanced feature selection accessible and practical
for various machine learning applications.
"""

import pandas as pd
from FSRLearning import Feature_Selector_RL

australian_data = pd.read_csv('australian_data.csv', header=None)

#DataFrame with the features
X = australian_data.drop(14, axis=1)

#DataFrame with the labels
y = australian_data[14]

fsrl_obj = Feature_Selector_RL(feature_number=14, nb_iter=100)
results = fsrl_obj.fit_predict(X, y)
fsrl_obj.compare_with_benchmark(X, y, results)

