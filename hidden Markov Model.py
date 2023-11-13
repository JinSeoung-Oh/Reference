# from https://towardsdatascience.com/hidden-markov-models-explained-with-a-real-life-example-and-python-code-2df2a7956d65

## Hidden Markov Models(HMM)
# Statistical models that work as a sequence of labeling problems. 
# These are the types of problems that describe the evolution of observable events, which themselves, 
# are dependent on internal factors that can’t be directly observed 

# 1. The invisible process is a Markov Chain, 
#    like chaining together multiple hidden states that are traversed over time in order to reach an outcome
#    Hidden Markov Models describe the evolution of observable events, which themselves, 
#    are dependent on internal factors that can’t be directly observed
#    None of the previous history of states you’ve been in the past matters to understand where you’re going next.
#    Markov Assumption, indicating that the probability of reaching the next state is only dependent on the probability of the current state.
#    - Likelihood or Scoring, as in, determining the probability of observing a sequence
#    - Decoding the best sequence of states that generated a specific observation
#    - Learning the parameters of the HMM that led to observing a given sequence, that traversed a specific set of states.

# In order to build a HMM that models in the training evaluation you need : 
#   1. Hidden States
#      Non-observable factors that influence the observation sequence
#   2. Transition Matrix
#      What’s the probability of going from one state to another. This matrix must also be row stochastic meaning 
#      that the probabilities from one state to any other state in the chain, each row in the matrix, must sum to one.
#   3. Sequence of Observations
#      Each observation representing the result of traversing the Markov Chain. Each observation is drawn from a specific vocabulary.
#   4. Observation Likelihood Matrix
#      Which is the probability of an observation being generated from a specific state.
#   5. Initial Probability Distribution
#      This is the probability that the Markov Chain will start in each specific hidden state.
#      There can also be some states will never be the starting state in the Markov Chain. In these situations, their initial probability is zero
#  The Initial Probability Distribution, along with the Transition Matrix and the Observation Likelihood, make up the parameters of an HMM

## The Forward Algorithm
# At each step you would take the conditional probability of observing the current outcome given that you’ve observed the previous outcome 
# and multiply that probability by the transition probability of going from one state to the other
# P(O) = sigma(Q) P(O,Q) = sigma(Q) P(O|Q) x P(Q)
# Observed sequence of scores = sum(hidden state sequene of joint pro. of observing a particluar hidden state sequence)
# Instead of computing the probabilities of all possible paths that 
# form that sequence the algorithm defines the forward variable and calculates its value recursively
# - alph(n,i) = P(x,...x_n, y_n=i|R) 
#   n = n_th outcome in the sequence of observed outcomes x / i = current Hidden state 
#   (y_n = i) = Hidden state i in the sequence of hidden states y / R = HMM parameters
# - alph(n,i) = sigma (k) [alph(n-1, k)t(k,i)P(x_n=x|y_n=i)]
#   n-1 = previous step in recursion / t(k,i) = Transition prob. from k to i /P(x_n=x|y_n=i) = prob. the n_th observaion will be x

## The Viterbi Algorithm
# Thinking in pseudo code, If you were to brute force your way into decoding the sequence of hidden states 
# that generate a specific observation sequence, all you needed to do was
#   1. generate all possible permutations of paths that lead to the desired observation sequence
#   2. use the Forward Algorithm to calculate the likelihood of each observation sequence, for each possible sequence of hidden states
#   3. pick the sequence of hidden states with highest probability
# - v_t(j) = max_(q1,...,qt)P(q1q2..q_t-1, O1O2..O_t-1, qt=j|R)
#   v_t(j) = viterbi path to hidden state j / max_(q1,...,qt) = indicates the algorithm is looking for the most probable path
#   q1q2..q_t-1 = hidden path sequence until the last time step / O1O2..O_t-1 = Observation sequence until the last time step
#   (qt = j) = current hidden state / R = HMM parameters
# The Viterbi path, the most probable path, is the path that has highest likelihood, from all the paths that can lead to any given hidden state.

## The Viterbi Algorithm in Python
from hmmlearn import hmm
import numpy as np

## Part 1. Generating a HMM with specific parameters and simulating the exam
print("Setup HMM model with parameters")
# init_params are the parameters used to initialize the model for training
# s -> start probability
# t -> transition probabilities
# e -> emission probabilities
model = hmm.CategoricalHMM(n_components=2, random_state=425, init_params='ste')

# initial probabilities
# probability of starting in the Tired state = 0
# probability of starting in the Happy state = 1
initial_distribution = np.array([0.1, 0.9])
model.startprob_ = initial_distribution

print("Step 1. Complete - Defined Initial Distribution")

# transition probabilities
#        tired    happy
# tired   0.4      0.6
# happy   0.2      0.8

transition_distribution = np.array([[0.4, 0.6], [0.2, 0.8]])
model.transmat_ = transition_distribution
print("Step 2. Complete - Defined Transition Matrix")

# observation probabilities
#        Fail    OK      Perfect
# tired   0.3    0.5       0.2
# happy   0.1    0.5       0.4

observation_probability_matrix = np.array([[0.3, 0.5, 0.2], [0.1, 0.5, 0.4]])
model.emissionprob_ = observation_probability_matrix
print("Step 3. Complete - Defined Observation Probability Matrix")

# simulate performing 100,000 trials, i.e., aptitude tests
trials, simulated_states = model.sample(100000)

# Output a sample of the simulated trials
# 0 -> Fail
# 1 -> OK
# 2 -> Perfect
print("\nSample of Simulated Trials - Based on Model Parameters")
print(trials[:10])

## Part 2 - Decoding the hidden state sequence that leads
## to an observation sequence of OK - Fail - Perfect

# split our data into training and test sets (50/50 split)
X_train = trials[:trials.shape[0] // 2]
X_test = trials[trials.shape[0] // 2:]

model.fit(X_train)

# the exam had 3 trials and your dog had the following score: OK, Fail, Perfect (1, 0 , 2)
exam_observations = [[1, 0, 2]]
predicted_states = model.predict(X=[[1, 0, 2]])
print("Predict the Hidden State Transitions that were being the exam scores OK, Fail, Perfect: \n 0 -> Tired , "
      "1 -> Happy")
print(predicted_states)
