# In this project we are going to see if we can build and train model for y = 2x WHICH I KNOW ITS BETTER NOT TO USE MACHINE LEARNING FOR THIS PROBLEM BUT I WANT TO TRY IT

import torch
from torch import nn
import matplotlib.pyplot as plt

# LETS BUILD SOME DATA
start = 1
end = 101
step = 1
X = torch.arange(start, end , step)
y = 2 * X

# Lets seprate our data to test and train
train_split  = int(0.8 * len(X))

X_train , y_train = X[:train_split], y[:train_split]
X_test , y_test = X[train_split:], y[train_split:]

# Lets build a function to visualize our function
def plot_predictions(X_train = X_train,
                     y_train = y_train,
                     X_test = X_test,
                     y_test = y_test,
                     predictions = None):
    
    plt.figure(figsize=(10,7))

    plt.scatter(X_train, y_train, c = "b", s = 4, label = "Training data")

    plt.scatter(X_test, y_test, c = "g", s=4, label = "Testing data")

    if predictions is not None:
        plt.scatter(X_test, predictions, c = "r", s=4, label = "Predictions")

    plt.xlabel("Input (x)")

    plt.ylabel("Output (y)")

    plt.title("Model predictions")
    
    plt.legend(prop={"size": 14})
    
    plt.show()

# Lets see if our function work or not
plot_predictions()