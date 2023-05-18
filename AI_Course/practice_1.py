# In this project we are going to see if we can build and train model for y = 2x WHICH I KNOW ITS BETTER NOT TO USE MACHINE LEARNING FOR THIS PROBLEM BUT I WANT TO TRY IT

import torch
from torch import nn
import matplotlib.pyplot as plt

# LETS BUILD SOME DATA
start = 1
end = 101
step = 1
b = 2
X = torch.arange(start, end , step)
y = b * X

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
# plot_predictions()

# Lets build a model
class MultiplyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Parameter(torch.randn(1,
                                          dtype = torch.float,
                                          requires_grad=True))
    def forward(self, x: torch.Tensor):
        return self.b * x


model_0 = MultiplyModel()

# Lets define our optimizer and loss function
loss_function = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.0001)

# Training loop
train_loss_values = []
test_loss_values = []
epoch_count = []
epochs = 2000

for epoch in range(epochs):

    model_0.train()

    y_pred = model_0(X_train)

    loss = loss_function(y_pred, y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model_0.eval()

    with torch.inference_mode():
        test_predictions = model_0(X_test)
        test_loss = loss_function(test_predictions, y_test)
        if epoch%100==0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
            print(model_0.state_dict())


plot_predictions(predictions=test_predictions)