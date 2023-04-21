import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
# from torch.utils.data import Dataset, DataLoader


# Loss function the DRO code use for binary data
def dro_loss(logits, labels, p):
    """Compute the DRO loss for binary classification."""
    logits = torch.ravel(logits)
    loss = nn.BCEWithLogitsLoss(reduction='none')(logits, labels)
    worst_case_loss, _ = torch.topk(loss, int(p*len(loss)))
    return torch.mean(worst_case_loss)




# Set up random seed
seed = 1
rng = np.random.RandomState(seed)

n = 1000 # Number of samples
d = 10 # Dimension / Features

# Parameters
rho_train = 1
rho_test = -1

num_epochs = 1000

# Define the model architecture
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1) # Input is 10d and output is 1d
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.sigmoid(x)
        return x


# Define lambda function for DGP
beta_gen = lambda d : np.array([1]*(d//2 + 1) + [0]*(d//2 - 1))
sigmoid = lambda x :  1/(1 + np.exp(-x))

# Define the DGP
def generate_spurious_data_anticausal(n, d, rho=0): 
    xs = rng.uniform(0.0, 1.0, size=(n * d)).reshape(n, d)
    # noise = rng.normal(0.0, 0.5, size=n)
    beta = beta_gen(d)/np.sqrt(d)
  
    p_y_1_g_x = sigmoid(xs @ beta + rho) # p( y = 1 | X ) = sigma(beta^T x)

    ys = np.random.random(size=(n,)) < p_y_1_g_x # why does this sample from the bernoulli distirbution p( y = 1 | X )?

    p_z_1_g_x = sigmoid(rho*(2*ys - 1)) # p( z = 1 | X ) = sigma(beta_z^T x)
    
    zs = np.random.random(size=(n,)) < p_z_1_g_x 

    xs[:, -1] = zs # This make model perfect as X and Y is perfect related

    return xs, ys, zs

# Generate train set
X,Y,Z = generate_spurious_data_anticausal(n, d, rho=rho_train)
X = torch.FloatTensor(X) # 
Y = torch.LongTensor(Y)

# Model
model = MyModel() 

# Loss Function
criterion = dro_loss

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model using ERM
for epoch in range(num_epochs):
    # forward
    outputs = model(X)
    loss = criterion(outputs, Y, p=0.1)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % (num_epochs/10) == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, num_epochs, loss.item()))

# Generate an OOD test set
X_test,Y_test,Z_test = generate_spurious_data_anticausal(n, d, rho=rho_test)
X_test = torch.FloatTensor(X_test)
Y_test = torch.LongTensor(Y_test)

# predict the test set
with torch.no_grad():
    outputs = model(X_test)
    predicted = (outputs > 0.5).float() # This line make things sus

predicted = torch.ravel(predicted)

# calculate the accuracy of the model on the test data 
# The way I calculate is count how many correct prediction and divide the total
correct = (predicted == Y_test).sum().item()
total = Y_test.size(0)
print(correct, total)
accuracy = correct / total

# print the accuracy
print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))

