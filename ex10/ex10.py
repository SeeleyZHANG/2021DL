from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent','Mozilla/5.0')]
urllib.request.install_opener(opener)

import torch
import collections
import numpy as np
import random

# some parameters
batch_size = 256
learn_rate = 1e-3
# number of characters in context 上下文字符数
context = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

# read training data from file
dataset = open("/Users/xinyi/Desktop/2021Spring/Deep Learning/code/shakespeare.txt")

# count number of different letters in text corpus
characters = set() #set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等
for line in dataset:  # 读每一行
    for a in line.lower().rstrip(): #rstrip() 删除 string 字符串末尾的指定字符（默认为空格
    # transform character into its ordinal value 
        characters.add(ord(a)) #以一个字符（长度为1的字符串）作为参数，返回对应的 ASCII 数值，或者 Unicode 数值
characters = sorted(characters) #对所有可迭代的对象进行排序操作，sort 是应用在 list 上的方法，sorted 可以对所有可迭代的对象进行排序操作。
D = len(characters)

# create association between characters and class indexes
C = {c:i for i,c in enumerate(characters)}
# 结果{32: 0, 33: 1, 39: 2, 40: 3, 41: 4...}
# create one-hot vectors for all characters in the batch
def one_hot_for(b):
    x = torch.zeros((b.shape[0],b.shape[1],D),device=device)
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
        # ignore unknown values (zero-padding)
            if b[i,j] >=0:
                x[i,j,b[i,j]] = 1
    return x

# implentation of recurrent network
# implentation of recurrent network
class Recurrent(torch.nn.Module):
    def __init__(self):
        super(Recurrent, self).__init__()
        # weight matrices with 1000 features
        self.W1 = torch.nn.Linear(in_features=D, out_features=1000)
        self.W2 = torch.nn.Linear(in_features=self.W1.out_features, out_features=D)
        # recurrent matrix
        self.Wr = torch.nn.Linear(in_features=self.W1.out_features,out_features=self.W1.out_features)
        self.activation = torch.nn.PReLU()

    def forward(self,x):
        # initialize hidden vector to zeros
        h_s = torch.zeros(len(x),self.W1.out_features, device=device)
        # remember all logit values
        Z = []
        for s in range(context): # 规定的长度
            # compute activation
            a_s = self.W1(x[:,s]) + self.Wr(h_s)  # apply activation function
            h_s = self.activation(a_s)
            # append logit values for current characters
            Z.append(self.W2(h_s))
        # return all logits for all characters
        return torch.stack(Z).transpose(1,0)
    
    # predict next character for the given sequence
    def predict(self,x):
      # initialize hidden vector to zeros
      h_s = torch.zeros(self.W1.out_features, device=device)
      for s in range(x.shape[1]):
        # compute activation
        a_s = self.W1(x[:,s]) + self.Wr(h_s)
        # apply activation function
        h_s = self.activation(a_s)
      # return logits for the last character only
      return self.W2(h_s)
    
def train_network():
    network = Recurrent().to(device)
    # cross-entropy loss
    loss = torch.nn.CrossEntropyLoss(ignore_index= -1)
    # SGD
    optimizer = torch.optim.SGD(network.parameters(), 
                                lr = learn_rate,
                                 momentum=0.9)

    # load training data
    dataset = open("/Users/xinyi/Desktop/2021Spring/Deep Learning/code/shakespeare.txt")

    #create dataset tensor
    # fixed context size
    data = collections.deque(maxlen=context)
    # instantiate with -1 (zero-padding)
    data.extend([-1] * context)
    X,T = [],[]
    for line in dataset:
        if not line.rstrip():
        # skip empty lines
            continue
        # iterate through all characters
        for a in line.replace("\n"," ").lower():
            # current input
            X.append(np.array(data))
            # append current character
            data.append(C[ord(a)])
            # current output
            T.append(np.array(data))

    print(f"Created dataset of {len(T)} samples with input size {context}x{D}")

    # dataset and data loader from tensor (use shuffling )
    DS = torch.utils.data.TensorDataset(torch.tensor(X), torch.tensor(T, dtype=torch.long))
    DL = torch.utils.data.DataLoader(DS, batch_size=batch_size, shuffle=True)

    try:
        # train for 100 epochs
        for epoch in range(100):
            # measure training loss (no validation set)
            total_loss = 0.
            for x,t in DL:
                # train network
                t = t.to(device)
                optimizer.zero_grad()
                # forward pass
                z = network(one_hot_for(x))
                # compute average loss
                J = torch.stack([loss(z[:,s], t[:,s]) for s in range(context)]).sum()
                J.backward()
                optimizer.step()
                # add up total loss
                total_loss += J.cpu().detach().item()
                print(f"\rloss: {float(J)/t.shape[0]: 3.5f}", end="")

            print(f"\rEpoch: {epoch} -- Loss: {total_loss/len(DS)}")
            # save model after each epoch(no validation)
            torch.save(network.state_dict(),"text.model")
            
    except KeyboardInterrupt:
        print()
    return network

def load_network():
    # load network from file
    network = Recurrent()
    network.load_state_dict(torch.load("text.model"))
    return network.to(device)

if __name__ == "__main__":
    # command line options (without argparse)
    import sys
    # first option : "train", "best"(get maximum character), other: sample character based on probabilities
    option = sys.argv[1] if len(sys.argv) > 1 else "best"
    # other options: seeding values for text
    samples = sys.argv[2:] if len(sys.argv) > 2 else ("the ", "beau", "mothe", "bloo")

    if option == "train":
        network = train_network()

    else:
        network = load_network()

        # go through all seeds

        for seed in samples:
            text = seed

            with torch.no_grad():
                # add 80 characters
                for i in range(80):
                    # turn current text to onr-hot batch
                    x = one_hot_for(np.array([[C[ord(s)] for s in text]]))
                    # predict the next character
                    z = network.predict(x)
                    # compute probabilities
                    y = torch.softmax(z,1).cpu().numpy()
                    if option == "best":
                        # take character with maximum probability
                        next_char = characters[np.argmax(y)]
                    else:
                        # sample character based on probabilities
                        next_char = random.choice(characters,y[0])[0]
                    # append character to text
                    text = text + char(next_char)

                # print seed and text
                print(f"{seed} -> \"{text}\"")