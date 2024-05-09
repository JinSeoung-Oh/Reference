from kan import KAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import numpy as np

dataset = {}
train_input, train_label = make_moons(n_samples=10000, shuffle=True, noise=0.1, random_state=None)
test_input, test_label = make_moons(n_samples=10000, shuffle=True, noise=0.1, random_state=None)

dataset['train_input'] = torch.from_numpy(train_input)
dataset['test_input'] = torch.from_numpy(test_input)
dataset['train_label'] = torch.from_numpy(train_label)
dataset['test_label'] = torch.from_numpy(test_label)

X = dataset['train_input']
y = dataset['train_label']
plt.scatter(X[:,0], X[:,1], c=y[:])

model = KAN(width=[2,2], grid=3, k=3) #KAN with two input and 2 output neurons

def train_accuracy():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_accuracy():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_accuracy, test_accuracy), loss_fn=torch.nn.CrossEntropyLoss())

formula1, formula2 = model.symbolic_formula()[0]

print(formula1)
#1012.55*sqrt(0.6*x_2 + 1) + 149.83*sin(2.94*x_1 - 1.54) - 1075.87

print(formula2)
#-948.72*sqrt(0.63*x_2 + 1) + 157.28*sin(2.98*x_1 + 1.59) + 1010.69

def acc(formula1, formula2, X, y):
    batch = X.shape[0]
    correct = 0
    for i in range(batch):
        logit1 = np.array(formula1.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)
        logit2 = np.array(formula2.subs('x_1', X[i,0]).subs('x_2', X[i,1])).astype(np.float64)
        correct += (logit2 > logit1) == y[i]
    return correct/batch

print('Training accuracy of the formula:', acc(formula1, formula2, dataset['train_input'], dataset['train_label']))
#Training accuracy of the formula: tensor(1.)

print('Testing accuracy of the formula:', acc(formula1, formula2, dataset['test_input'], dataset['test_label']))
#Testing accuracy of the formula: tensor(0.9990)
