import torch
from torch.nn.modules import transformer
import torchvision
import random
import numpy
from torchvision.transforms.transforms import ToTensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RBF(torch.nn.Module):
    def __init__(self, no_means, sample_dim):
        super(RBF, self).__init__()

        self.D = sample_dim # implement for batch
        self.K = no_means
        self.W = torch.nn.Parameter(torch.zeros(no_means,sample_dim))
        torch.nn.init.normal_(self.W,0,1)
        self.var = torch.nn.Parameter(torch.ones(no_means))


    def forward(self,x):
        B = x.shape[0]
        W = self.W.unsqueeze(0).expand(B,self.K,self.D)
        X = x.unsqueeze(1).expand(B,self.K,self.D)
        A = torch.sum(torch.pow(W-X,2), -1) # activation function

        result = torch.exp(-A/self.var)

        return result

class Network(torch.nn.Module):
    def __init__(self,no_hidden,no_means):
        super(Network,self).__init__()

        # define network structure
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32,kernel_size=(5,5),stride=1,padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels,out_channels=64,kernel_size=(5,5),stride=1,padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
        self.activation = torch.nn.Sigmoid()
        self.bn = torch.nn.BatchNorm2d(self.conv2.out_channels)
        self.fc1 = torch.nn.Linear(7*7*64, no_hidden,bias=True)
        self.rbf = RBF(no_means,no_hidden)
        self.fc2 = torch.nn.Linear(no_means,10,bias=True)

    def forward(self,x):
        # forward through layers
        a = self.extract(x)
        return self.fc2(self.rbf(a))

    def extract(self,x):
        # forward through layers
        a = self.activation(self.pool(self.conv1(x)))
        a = self.activation(self.bn(self.pool(self.conv2(a))))
        # a = self.activation(self.pool(self.conv2(a)))
        a = torch.flatten(a,1)
        return self.fc1(a)

# training set :MNIST

train_set = torchvision.datasets.MNIST(
    root = "temp",
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

# data loader with batch size 32
train_loader = torch.utils.data.DataLoader(
    train_set,
    shuffle = True,
    batch_size = 32
)

# test set: MNIST
test_set = torchvision.datasets.MNIST(
    root = "temp",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()

)
test_loader = torch.utils.data.DataLoader(
    test_set,
    shuffle=False,
    batch_size=32
)

# create network
network = Network(2,100)

if __name__ == "__main__":
    network = network.to(device)

    # Cross-entropy loss with softmax activation
    loss = torch.nn.CrossEntropyLoss()
    # stochastic gradient descent
    optimizer = torch.optim.SGD(
        params=network.parameters(),
        lr = 1e-4, momentum=0.9
    )

    # train network for a few epochs
    best = 0
    torch.save(network.state_dict(), F"Init.model")

    for epoch in range(1000):
        # iterate over training batches
        for x, t in train_loader:
            optimizer.zero_grad()
            # forward batch through network nad obtain logits
            z = network(x.to(device))
            # compute the loss
            J = loss(z, t.to(device))
            # compute the gradient via automatic differentiation
            J.backward()
            # perform weight update
            optimizer.step()
        
        # compute test accuracy
        correct = 0
        with torch.no_grad():
            for x,t in test_loader:
                # compute logits for batch
                z = network(x.to(device))
                # compute the index of the largest logits per sample
                _, y = torch.max(z.data, 1)
                # compute how often the correct index was predicted
                correct += (y == t.to(device)).sum().item()
          
          # print(network.rbf.var)
        
        # print epoch and accuracy
        print(F"Epoch {epoch+1}; test accuracy: {correct/len(test_set)*100.:1.2f}%")
        if correct > best:
            best = correct
            torch.save(network.state_dict(), F"Best.model")
    
    print()

