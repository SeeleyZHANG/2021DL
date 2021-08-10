from six.moves import urllib
opener = urllib.request.build_opener()
opener.addHeaders = [('User-agent','Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
import torchvision
import numpy
train_adversatial = False
train_with_noise = False

filename = f"Results_{'noise' if train_with_noise else 'adv' if train_adversatial else 'none'}_bn.txt"

torch.manual_seed(42)

# network implementation
class Convolutional(torch.nn.Module):
    def __init__(self, K, O):
    # call base calss constructor
        super(Convolutional, self).__init__()
        # some convolutional layers
        self.conv1 = torch.nn.Conv2d(in_channels=1,out_channels=32,kernel_size=(5,5),stride=1,padding=2)
        self.conv2 = torch.nn.Conv2d(in_channels=self.conv1.out_channels,out_channels=32,kernel_size=(5,5),stride=1,padding=2)
        self.pool = torch.nn.MaxPool2d(kernel_size=(2,2),stride=2)
        
        # sigmoid activation
        self.activation = torch.nn.Sigmoid()
        # batch norm
        self.bn = torch.nn.BatchNorm2d(self.conv2.out_channels)
        # some fully-connect layers
        self.fc1 = torch.nn.Linear(7*7*32,K,bias=True)
        self.fc2 = torch.nn.Linear(K,O)

    # forward propagation through the netword
    def forward(self,x):
      # first layer convolution 
        a = self.activation(self.pool(self.conv1(x)))
        # second layer convolution and BN
        #a = self.activation(self.bn(self.pool(self.conv2(a))))
        a = self.activation(self.pool(self.conv2(a)))
        # dully-connect layers
        a = torch.flatten(a,1)
        return self.fc2(self.activation(self.fc1(a)))
# training and test set
train_data = torchvision.datasets.MNIST(
    root="temp",
    train = True,
    download = True,
    transform = torchvision.transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    train_data,
    shuffle=True,
    batch_size=128
)
test_data = torchvision.datasets.MNIST(
    root="temp",
    train = False,
    download = True,
    transform = torchvision.transforms.ToTensor()
)
test_loader = torch.utils.data.DataLoader(
    test_data,
    shuffle=False,
    batch_size=100
)


# run on cude device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create network with 59 hidden neurons in FC layer
network = Convolutional(50,10).to(device)

# select loss function; all three should work
loss = torch.nn.CrossEntropyLoss()

# SGD optimizer
optimizer = torch.optim.SGD(
    params=network.parameters(),
    lr=1e-2,
    momentum=0.9
)
# generate FGS adversarial samples
def FGS(x, t, alpha=0.3):
    # tell aurograd that we need to gradient for the input
    x.requires_grad_(True) # enable gradient for input
    # make sure that there are no remainings
    network.zero_grad()
    # forward input
    z = network.forward(x)
    # compute loss and gradient
    J = loss(z,t)
    J.backward()

    # get the gradient
    grad = x.grad.detach()
    # perform gradient ascent
    return torch.clamp(x+alpha*torch.sign(grad),0,1)

def noise(x,alpha=0.3):
    noise = torch.bernoulli(torch.ones(x.shape)*0.5)*2-1
    return torch.clamp(x+alpha*noise.to(device),0,1)

# train serveral epoachs
acc_clean, acc_adv = [],[]
try:
  for epoch in range(50):
    for x,t in train_loader:
      optimizer.zero_grad()
      x,t = x.to(device), t.to(device)
      # compute output for current batch
      z = network(x)
      # compute loss
      J = loss(z,t)
      # compute gradient
      J.backward()

      if train_adversatial:
        # compute adversarials for batch
        x_hat = noise(x) if train_with_noise else FGS(x,t)
        # compute output for adversarials
        z_hat = network(x_hat)
        # compute loss for adversarials
        J = loss(z_hat,t)
        # computer gradient
        J.backward()

      # perform combined optimizer step
      optimizer.step()

    # evaluation
    correct_clean=0
    correct_adv=0
    for x,t in test_loader:
      x,t = x.to(device), t.to(device)
      with torch.no_grad():
        z = network(x)
        # computer classification accuracy
        correct = torch.argmax(z,dim=1) == t
        correct_clean += torch.sum(correct)

      # create adversarial samples for correctly classified sample
      x = x[correct]
      t = t[correct]
      x_hat = FGS(x,t)

      with torch.no_grad():
        z_hat = network(x_hat.to(device))
        # computer classification accuracy
        correct = torch.argmax(z_hat,dim=1) == t
        correct_adv += torch.sum(correct)

      acc_clean.append(correct_clean/len(test_data))
      acc_adv.append(correct_adv/correct_clean)
      print(f"Epoch {epoch+1}:"
            f"Clean accuracy: {correct_clean}/{len(test_data)} = {correct_clean/len(test_data)*100:3.2f}%; "
            f"Adver accuracy: {correct_adv}/{correct_clean} = {correct_adv/correct_clean*100:3.2f}%")

except KeyboardInterrupt:
  pass

  # write accuracies to fils

  with open(filename, "w") as f:
      f.write(",".join(str(float(v))for v in acc_clean))
      f.write("\n")
      f.write(",".join(str(float(v))for v in acc_adv))