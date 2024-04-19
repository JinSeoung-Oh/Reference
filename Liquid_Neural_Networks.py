"""
# Liquid_Neural_Network
A time-continuous Recurrent Neural Network(RNN) that processes data sequentially, keeps the memory of past inputs, adjusts its behaviors based on new inputs
and can handle variable-length inputs to enhacne the task-understading capailties of NNs

1. Dynamic architecture
Its neurons are more expressive than the neurons of a regular neual network, making LNNs more interpretable. They can handle real-time sequential data effectively

2. Continual leaning & adaptability
LNNs adapt to chaning data enve after traning, mimicking the brain of living oranisms(?) more accurately compared to traditional NNs that stop learning new information after the model training phase
Hence, LNNs don't require vast amounts of labeled training data to generate accurate results.

Since LNN neurons offer rich connections that can express more informtation, they are smaller in size compared to regular NNs. Hence, it becomes easier for researchers to explain how an LNN reached a decision.

Ex 1. Time Series data processing & forecasting
Ex 2. Image & video processing
Ex 3. Natual Language understaind

# Contraints & Challenges of Liquid NN
1. Vanishing Gradinet Problem
2. Parameter Tuning
3. Lack of Literature
"""

# From https://python.plainenglish.io/liquid-neural-networks-simple-implementation-395e43879060

import numpy as np
import tensorflow as tf

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define your Liquid Neural Network (LNN) class
class LiquidNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights, biases, time constants, etc.
        self.W_in = np.random.randn(input_size, hidden_size)
        self.W_hid = np.random.randn(hidden_size, hidden_size)
        self.W_out = np.random.randn(hidden_size, output_size)
        self.bias_hid = np.zeros(hidden_size)
        self.bias_out = np.zeros(output_size)
        self.time_constant = 0.1  # Adjust as needed

    def forward(self, x):
        # Implement the dynamics (e.g., Euler integration)
        hidden_state = np.zeros(self.W_hid.shape[1])
        outputs = []

        for t in range(len(x)):
            hidden_state = (1 - self.time_constant) * hidden_state + \
                            self.time_constant * np.dot(x[t], self.W_in) + \
                            np.dot(hidden_state, self.W_hid) + self.bias_hid
            output = np.dot(hidden_state, self.W_out) + self.bias_out
            # Apply activation function (e.g., sigmoid)
            exp_output = np.exp(output)
        softmax_output = exp_output /
            output.append(exp_output)

        return np.array(outputs)

# Example usage with CIFAR-10 data
input_size = 32 * 32 * 3  # Input size for CIFAR-10 images
hidden_size = 20
output_size = 10  # Number of classes in CIFAR-10
net = LiquidNeuralNetwork(input_size, hidden_size, output_size)

# Use the training data (x_train) as your input
predictions = net.forward(x_train)

