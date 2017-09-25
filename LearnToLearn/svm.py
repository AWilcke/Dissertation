import torch
import torch.nn as nn
from torch.autograd import Variable

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

        print("Training...")

        for epoch in range(self.max_epochs):
            shuffle = torch.randperm(total_count)

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
                loss.backward()
                optimiser.step()
            print("Epoch {}".format(epoch))

        self.w = self.state_dict()['fc.weight'].cpu().numpy().reshape(4096)
