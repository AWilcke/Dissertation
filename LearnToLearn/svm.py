import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class LinearSVM(nn.Module):
    
    def __init__(self, D_in, lr=0.1, c=1e-2, max_epochs=100, batch_size=16):
        super(LinearSVM, self).__init__()
        self.fc = nn.Linear(D_in, 1)
        self.batch_size = batch_size
        self.lr = lr
        self.c = c
        self.max_epochs = max_epochs
        self.w = None

    def forward(self, x):
        h = self.fc(x)
        return h

    def fit(self, X, Y):
        
        self.train()
        if torch.cuda.is_available():
            self = self.cuda()

        X = torch.from_numpy(X).float()
        Y = torch.from_numpy(Y).float()
        total_count = Y.shape[0]

        optimiser = torch.optim.SGD(self.parameters(), lr=self.lr)


        for epoch in range(self.max_epochs):
            shuffle = torch.randperm(total_count)
            total_loss = 0

            for i in range(0, total_count, self.batch_size):
                x = X[shuffle[i:i+self.batch_size]]
                y = Y[shuffle[i:i+self.batch_size]]

                x = Variable(x)
                y = Variable(y)

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                
                optimiser.zero_grad()
                output = self(x)
                
                loss = torch.mean(torch.clamp(1 - output * y, min=0))  # hinge loss
                loss += self.c * torch.mean(self.fc.weight**2)  # l2 penalty
                total_loss += loss.data.cpu().numpy()
                loss.backward()
                optimiser.step()

            print("{0:.1f}%".format(epoch/self.max_epochs*100), end='\r')

        if torch.cuda.is_available():
            self = self.cpu()

        self.w = self.state_dict()['fc.weight'].numpy().reshape(self.fc.in_features)

    def predict(self, X):
        X = torch.from_numpy(X).float()
        X = Variable(X)
        predict = self(X).data.numpy()
        predict[predict<0] = -1
        predict[predict>=0] = 1
        return predict.reshape(-1)
    
    def eval(self, X, Y):
        x_predict = self.predict(X)
        return np.count_nonzero(x_predict == Y)/len(Y)
