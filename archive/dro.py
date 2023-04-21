import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Loss function the groupDRO code use for binary data
def hinge_loss(yhat, y):
    # The torch loss takes in three arguments so we need to split yhat
    # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
    # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
    # so we need to swap yhat[:, 0] and yhat[:, 1]...
    torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
    y = (y.float() * 2.0) - 1.0
    return torch_loss(yhat[:, 1], yhat[:, 0], y)

# Define the DRO objective function
def dro_loss(y_pred, y_true, epsilon):
    # Compute the worst-case loss over a set of distributions
    worst_case_loss = 0
    for p in range(len(y_true)):
        p_hat = y_true[p] / len(y_true)
        loss = hinge_loss(reduction='none')(y_pred, y_true[p].long())
        worst_case_loss += torch.sum(p_hat * torch.exp(-epsilon * loss))
    return worst_case_loss

# Define the binary classification model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Lambda function for sigmoid function and beta generation (half 0, half 1)
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

n = 100 # number of samples
d = 10 # dimension

seed = 1
rng = np.random.RandomState(seed)

# Generate some sample data
X = rng.rand(n, d)

y = rng.randint(2, size=100)

# Convert the data to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# Split the data into training and validation sets
X_train, X_val = X[:80], X[80:]
y_train, y_val = y[:80], y[80:]

# Define the DRO hyperparameters
epsilon = 0.1
batch_size = 10

# Instantiate the model and optimizer
model = BinaryClassifier()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model using DRO
for epoch in range(100):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        
        optimizer.zero_grad()
        y_pred = model(batch_X)
        print(y_pred.shape, batch_y.shape)
        loss = dro_loss(y_pred, batch_y, epsilon)
        loss.backward()
        optimizer.step()

    # Compute the validation loss
    y_pred_val = model(X_val)
    loss_val = dro_loss(y_pred_val, y_val, epsilon)

    print('Epoch {}, Validation Loss: {:.4f}'.format(epoch+1, loss_val.item()))
