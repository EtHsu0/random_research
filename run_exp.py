import numpy as np
import torch 
import argparse
import dgp
import loss
import models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
rho = 1

def split_group(X, Y, Z, dim):
    g1 = np.empty((0, dim))
    g2 = np.empty((0, dim))
    g3 = np.empty((0, dim))
    g4 = np.empty((0, dim))

    for idx, (y_val, z_val) in enumerate(zip(Y, Z)):
        if y_val == 0 and z_val == 0:
            g1 = np.vstack((g1, X[idx]))
        elif y_val == 0 and z_val == 1:
            g2 = np.vstack((g2, X[idx]))
        elif y_val == 1 and z_val == 0:
            g3 = np.vstack((g3, X[idx]))
        elif y_val == 1 and z_val == 1:
            g4 = np.vstack((g4, X[idx]))

    return g1,g2,g3,g4

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['Linear', 'NeuralNet'], default="NeuralNet")
    parser.add_argument('-o', '--optim', choices=['ERM', 'worstLoss', 'groupDRO'], default="ERM")
    parser.add_argument('--distr', choices=['IID', 'OOD'], default='OOD')
    parser.add_argument('--DGP', choices=['anticasual', 'linear', 'plusminus', 'fortest', 'normal'], default='anticasual')
    parser.add_argument('-n', '--num_samples', type=int, default=10000)
    parser.add_argument('-d', '--dim', type=int, default=100)
    parser.add_argument('-e', '--num_epochs', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    n = args.num_samples
    d = args.dim

    if args.model == "Linear":
        model = models.linearModel(d, 1)
    elif args.model == "NeuralNet":
        model = models.NeuralNetwork(d, 1)
    # 
    # if args.optim == "ERM":
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    # elif args.optim == "worstLoss":
        # criterion = nn.BCELoss()
        # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

    if args.distr == "IID":
        X_test,Y_test,Z_test = dgp.generate_DGP(n//5, d, rho, args.DGP)
    elif args.distr == "OOD":
        X_test,Y_test,Z_test = dgp.generate_DGP(n//5, d, -rho, args.DGP)
    

    X_test = torch.FloatTensor(X_test)
    Y_test = torch.FloatTensor(Y_test).ravel()

    X,Y,Z = dgp.generate_DGP(n, d, rho, args.DGP)

    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(Y).ravel()
    g1,g2,g3,g4 = split_group(X,Y,Z, d)
    groups = [g1,g2,g3,g4]
    labels = []
    for idx, g in enumerate(groups):
        groups[idx] = torch.FloatTensor(g)
        if (idx < 2):
            tensor = torch.zeros(len(groups[idx]))
        else:
            tensor = torch.ones(len(groups[idx]))
        labels.append(tensor)

    losses = []
    epoches = []
    losses_test = []

    nq = 0.1
    # ntheta = 0.1
    q = np.ones(4) # Four weights
    q = q / np.sum(q)
    theta = np.random.randn(100) #???
    for epoch in range(args.num_epochs):
        #     # print(value.shape)
        if args.optim == "worstLoss":
            loss = None
            for idx, value in enumerate(groups):
                outputs = model(value)
                group_loss = criterion(outputs.squeeze(), labels[idx])
                if loss == None:
                    loss = group_loss
                elif loss.item() < group_loss.item():
                    loss = group_loss
        elif args.optim == "ERM":
            outputs = model(X)
            loss = criterion(outputs.squeeze(), Y)
        else:
            group_id = np.random.randint(4)
            outputs = model(groups[group_id])
            group_loss = criterion(outputs.squeeze(), labels[group_id])

            q[group_id] = q[group_id] * np.exp(group_loss.item() * nq)
            q = q / np.sum(q)
            model.zero_grad()
            optimizer.zero_grad()
            loss = group_loss * q[group_id]
            # group_loss.backward()
            # optimizer.step()

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if ((epoch + 1) % (args.num_epochs / 10) == 0) or epoch == 0:
            epoches.append(epoch)
            with torch.no_grad():
                outputs = model(X)
                loss = criterion(outputs.squeeze(), Y)
                losses.append(loss.item())
                
                outputs_test = model(X_test)
                predicted = (outputs_test > 0.5).float()
                loss_test = criterion(outputs_test.squeeze(), Y_test)
                correct = (predicted.squeeze() == Y_test).sum().item()
                total = Y_test.size(0)
                accuracy = correct / total
                losses_test.append(loss_test)
                print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.num_epochs, loss.item()))
                # print the accuracy
                print('\tAccuracy on test data: {:.2f}%'.format(accuracy * 100))
                print('\tLoss_test: {:.4f}'.format(loss_test.item()))
                
                
    
    plt.plot(epoches, losses, label="train")
    plt.plot(epoches, losses_test, label="test")
    plt.legend()
    plt.savefig(f'images/{args.model}_{args.optim}_{args.DGP}_{args.num_epochs}_{args.num_samples}_{args.dim}.png', bbox_inches='tight')
    plt.show()

    # with torch.no_grad():
    #     outputs = model(X_test)
    #     predicted = (outputs > 0.5).float()

    # correct = (predicted.squeeze() == Y_test).sum().item()
    # predicted_test = (X_test[:, -1] > 0.5).float()
    # correct_test = (predicted_test.squeeze() == Y_test).sum().item()
    # print("Number of correct prediction using last feature of X:", correct_test)
    # total = Y_test.size(0)
    # print("Correct and total ratio of the prediction:", correct, total)
    # accuracy = correct / total

    # # print the accuracy
    # print('Accuracy on test data: {:.2f}%'.format(accuracy * 100))


if __name__=='__main__':
    main()
