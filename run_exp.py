import numpy as np
import torch 
import argparse
import dgp
import models
import torch.nn as nn
import torch.optim as optim

rho = 1

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', choices=['ERM', 'DRO', 'groupDRO', 'TEST'], default="ERM")
    parser.add_argument('--distr', choices=['IID', 'OOD'], default='IID')
    parser.add_argument('--DGP', choices=['anticasual', 'linear', 'plusminus', 'fortest', 'normal'], default='anticasual')
    parser.add_argument('-n', '--num_samples', type=int, default=1000)
    parser.add_argument('-d', '--dim', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.num_samples
    d = args.dim
    X,Y,Z = dgp.generate_DGP(n, d, rho, args.DGP)
    
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y).ravel()

    if args.model == "ERM":
        model = models.ERMModel()
        criterion = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(args.num_epochs):
        outputs = model(X)

        loss = criterion(outputs.squeeze(), Y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % (args.num_epochs / 10) == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch, args.num_epochs, loss.item()))

    if args.distr == "IID":
        X_test,Y_test,Z_test = dgp.generate_DGP(n//5, d, rho, args.DGP)
    elif args.distr == "OOD":
        X_test,Y_test,Z_test = dgp.generate_DGP(n//5, d, -rho, args.DGP)
        
    
    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test).ravel()

    with torch.no_grad():
        outputs = model(X_test)
        predicted = (outputs > 0.5).float()

    correct = (predicted.squeeze() == Y_test).sum().item()
    predicted_test = (X_test[:, -1] > 0.5).float()
    correct_test = (predicted_test.squeeze() == Y_test).sum().item()
    print("Number of correct prediction using last feature of X:", correct_test)
    total = Y_test.size(0)
    print("Correct and total ratio of the prediction:", correct, total)
    accuracy = correct / total

    # print the accuracy
    print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))


if __name__=='__main__':
    main()
