from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User=agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
import torchvision
import numpy

class TargetVector():
    def __init__(self, known_targets = (4,5,8,9), unknown_targets = (0,2,3,7), ignored_targets = (1,6)):
        # know target classes: get one-hot vectors
        self.known_targets = known_targets
        # unknown target classes: get 1/0 vectors
        self.unknown_targets = unknown_targets
        self.ignored_targets = ignored_targets
        # one-hot vectors for 0 classes
        # eye(): 可以用来构造单位矩阵
        self.one_hot_known = numpy.eye(len(known_targets))
        self.target_known = {k:self.one_hot_known[i] for i,k in enumerate(self.known_targets)}
        # 1/O-vectors for O classes 1/4 = 0.25
        self.target_unknown = numpy.ones(len(known_targets)) / len(known_targets)

    # creates the target batch for the given targets
    def __call__(self, inputs, targets):
        # split off unknown samples
        valid = []
        vectors = [] 
        for i, t in enumerate(targets):
            if t in self.known_targets:
                # append one-hot vector for target class
                vectors.append(self.target_known[int(t)])
                valid.append(inputs[i].numpy())
            elif i in self.unknown_targets:
                # append 1/O vector
                vectors.append(self.target_unknown)
                valid.append(inputs[i].numpy())
        
        # return the filtered original inputs and their one-hot vectors
        return torch.tensor(valid), torch.tensor(vectors)
    
    # computes the predicted class and the confidence for that class
    def predict(self, logits):
        # softmax over logits
        confidence = torch.nn.functional.softmax(logits, dim=1)
        # indexes of the prediction
        indexes = torch.argmax(logits, dim=1)
        # confidence values for the predicted classes
        max_confidences = confidence[range(len(logits)), indexes]
        # return tuple: (predicted class, confidence of that class) for samples in the batch
        return [(self.known_targets[indexes[i]], max_confidences[i]) for i in range(len(logits))]
    
    # computes the confidence metric for the given sample
    def confidences(self, logits, targets):
        # softmax over logits
        confidences = torch.nn.functional.softmax(logits, dim=1).numpy()
        # return confidence of correct class for known samples and 1-max(confidences) + 1/0 for unknown samples
        return [
                # known targets
                numpy.sum(confidences[i] * self.target_known[int(targets[i])])
                    if targets[i] in self.known_targets
                    # unknown targets
                    else 1 - numpy.max(confidences[i]) + 1./len(self.known_targets)
                # iterate over batch
                for i in range(len(logits))
        ]

# Compute the Softmax loss from logits and one-hot target vectors
def adapted_softmax_loss(self, logits, targets):
    # compute loss
    loss = - torch.mean(logits * targets) + torch.mean(torch.logsumexp(logits, dim=1)) / targets.shape[1]
    # loss = -torch.mean(torch.nn.functional.log_softmax(logits, dim=1) * targets)
    return loss

# Define autograd function for our loss implementation
class AdaptedSoftMaxFUnction(torch.autograd.Function):

    # implement the forward propagation
    @staticmethod
    def forward(ctx, logits, targets):
        # compute the log probabilities via log_softmax
        log_y = torch.log_softmax(logits, dim=1)
        # save log probabilities and targets for backward computation
        ctx.save_for_backward(log_y, targets)
        # compute loss
        loss = - torch.mean(log_y * targets)
        return loss

    # implement Jacobian
    @staticmethod
    def backward(ctx, result):
        # get results stored from forward pass
        log_y, targets = ctx.saved_tensors
        # compute probabilities from log probabilities
        y = torch.exp(log_y)
        # return y-t as Jacobian for the logits, None for the targets
        return y-targets, None

# Network Implementation
class Convolutional(torch.nn.Module):
    def __init__(self, K, O):
        # call base class constructor
        super(Convolutional,self).__init__()
        # some convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(5,5),stride=1,padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=32,kernel_size=(5,5),stride=1,padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size = (2,2), stride=2)
        # sigmoid activation
        self.activation = torch.nn.Sigmoid()
        # batch norm
        self.bn = torch.nn.BatchNorm2d(self.conv2.out_channels)
        # some fully-connected layers
        self.fc1 = torch.nn.Linear(7*7*32, K, bias=True)
        self.fc2 = torch.nn.Linear(K, O)
    
    # forward propagation through the network
    def forward(self, x):
        # first layer convolution
        a = self.activation(self.pool(self.conv1(x)))
        # second layer convolution and BN
        a = self.activation(self.bn(self.pool(self.conv2(a))))
        # fully-connected layers
        a = torch.flatten(a, 1)
        return self.fc2(self.activation(self.fc1(a)))


# training and test set
train_set = torchvision.datasets.MNIST(
    root = "temp",
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(
    train_set,
    shuffle = True,
    batch_size = 32
)

test_set = torchvision.datasets.MNIST(
    root = "temp",
    train = False,
    download = True,
    transform = torchvision.transforms.ToTensor()
)

test_loader = torch.utils.data.DataLoader(
    test_set,
    shuffle = False,
    batch_size = 32
)

# run the cuda device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create network with 50 hidden neurons in FC layer
network = Convolutional(50,4).to(device)

# select loss function; all three should work
# loss = AdapatedSoftMaxLoss()
loss = AdaptedSoftMaxFUnction.apply

# SGD optimizer
optimizer = torch.optim.SGD(
    params = network.parameters(),
    lr=0.001, momentum=0.9
)

# target vector implementation
targets = TargetVector()

threshold = 0.5

# train several epochs
for epoch in range(100):
    for x,t in train_loader:
        optimizer.zero_grad()
        # convert targets and filter unknown unknowns
        x,t = targets(x,t)
        z = network(x.to(device))
        # compute out loss
        J = loss(z, t.to(device))
        # perform update
        J.backward()
        optimizer.step()

# evaluation: correctly classified and total number of samples
k, ku, uu = 0, 0, 0
nk, nku, nuu = 0, 0, 0
# evaluation: average confidence
conf = 0.
with torch.no_grad():
    for x,t in test_loader:
        # compute network output
        z = network(x.to(device)).cpu()
        # compute predicted classes and their confidence
        predictions = targets.predict(z)
        # add confidence metric for batch
        conf += numpy.sum(targets.confidences(z,t))
        # compute accuracy
        for i in range(len(t)):
            # iterate over all samples in the batch
            if t[i] in targets.known_targets:
                # known sample: correctly classified?
                if predictions[i][0] == int(t[i]) and predictions[i][1] >= threshold:
                    k += 1
                nk += 1
            elif t[i] in targets.unknown_targets:
                # known unknown sample: correctly rejected
                if predictions[i][1] < threshold:
                    ku += 1
                nku += 1
            else:
                # unknown unknown sample: correctly rejected
                if predictions[i][1] < threshold:
                    uu += 1
                nuu += 1

# print epoch and metrics
print(F"Epoch {epoch}; test known: {k/nk*100.:1.2f} %, known unknown: {ku/nku*100.:1.2f} %, unknown unknown: {uu/nuu*100.:1.2f} %; average confidence: {conf/len(test_set):1.5f}")

print()
