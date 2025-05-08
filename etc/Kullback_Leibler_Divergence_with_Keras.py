### From https://medium.com/h7w/kullback-leibler-divergence-with-keras-227ef84f2a1b
"""
1. Framing the Problem: From Predictions to Loss Functions
   -a. In supervised learning, a model makes predictions in a feed-forward pass and these predictions are compared to 
       the true targets to guide optimization.
   -b. Choosing an appropriate loss function quantifies “how wrong” the model’s output is.
   -c. One such loss is Kullback–Leibler (KL) divergence, which measures the difference between two probability distributions.

2. From Entropy to KL Divergence
   Entropy measures the information content of a distribution.
   -a. Definition (Wikipedia, 2001):
       𝐻(𝑝)=−∑_𝑥 𝑝(𝑥) log 𝑝(𝑥)
   -b. Intuition:
       “The minimum number of bits it would take to encode our information” when using log_2
       As model weights change, the predicted distribution 𝑞 shifts. 
       To optimize, we need to quantify the information lost when approximating the true distribution 𝑝 by 𝑞

   KL divergence (or relative entropy) extends entropy to compare two distributions.
   -a. Expectation form:
       𝐷_(KL)(𝑝∥𝑞)=𝐸_(𝑥∼𝑝)[log 𝑝(𝑥)−log 𝑞(𝑥)]=∑_𝑥 𝑝(𝑥)(log 𝑝(𝑥)−log 𝑞(𝑥))
   -b. Ratio form (most common):
       𝐷_(KL)(𝑝∥𝑞)=∑_𝑥 𝑝(𝑥) log(𝑝(𝑥)/𝑞(𝑥))
   -c. Interpretation:
       The expected number of bits of information lost when using 𝑞 to approximate 𝑝
       
3. When to Use KL Divergence
   -a. Variational Autoencoders (VAEs)
       -1. Encoder maps inputs to a latent distribution, decoder reconstructs data.
       -2. KL divergence regularizes the learned latent distribution toward a chosen prior (e.g., standard normal), 
           enabling generative sampling.
   -b. Multi-Class Classification
       -1. Softmax outputs a 𝐾-class probability vector.
       -2. KL divergence compares the entire predicted distribution to the true one-hot distribution, not just the argmax.
   -c. Alternative to Least Squares in Regression
       -1. Traditional regression minimizes squared error (𝑦_pred−𝑦_true)^2
       -2. KL divergence can instead compare predicted and true output distributions, 
           offering robustness when prediction noise is non-Gaussian.
"""

import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

# Model configuration
img_width, img_height         = 32, 32
batch_size                    = 250
no_epochs                     = 25
no_classes                    = 10
validation_split              = 0.2
verbosity                     = 1

# Load CIFAR10 dataset
(input_train, target_train), (input_test, target_test) = cifar10.load_data()

# Reshape data based on channels first / channels last strategy.
# This is dependent on whether you use TF, Theano or CNTK as backend.
# Source: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py
if K.image_data_format() == 'channels_first':
    input_train = input_train.reshape(input_train.shape[0],3, img_width, img_height)
    input_test = input_test.reshape(input_test.shape[0], 3, img_width, img_height)
    input_shape = (3, img_width, img_height)
else:
    input_train = input_train.reshape(input_train.shape[0], img_width, img_height, 3)
    input_test = input_test.reshape(input_test.shape[0], img_width, img_height, 3)
    input_shape = (img_width  , img_height, 3)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data.
input_train = input_train / 255
input_test = input_test / 255

# Convert target vectors to categorical targets
target_train = keras.utils.to_categorical(target_train, no_classes)
target_test = keras.utils.to_categorical(target_test, no_classes)

# Create the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.50))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss=keras.losses.kullback_leibler_divergence,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Fit data to model
model.fit(input_train, target_train,
          batch_size=batch_size,
          epochs=no_epochs,
          verbose=verbosity,
          validation_split=validation_split
)

# Generate generalization metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')

