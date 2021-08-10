import numpy as np
from matplotlib import pyplot as plt

# sample 100 z-locationa in range [-2,2]
X = np.random.random(100) * 4 - 2
# good values for polynomial
#X = np.random.random(100) *8-4.5

# 1. cosine waves t = 1/2(cos(3x)+1)
cos = np.cos(X*3.) / 2. + 0.5

# 2. Gaussian bell curve
sigma = .5
gauss = np.exp(-(X**2) * (sigma**2))

# 3. Polynome(in range [-4.5,3.5])
poly = (X**5 + 3*X**4-11*X**3 - 27*X**2 + 10*X +64) / 100

# add x_0 to the input # ones()返回一个全1的n维数组
X = np.vstack((np.ones(X.shape), X)) #垂直（按照行顺序）的把数组给堆叠起来。

# 2-layer network with K hidden nodes
K = 4
eta = .1
T = cos

# randomly initialize weights
w1 = np.random.random((K+1,2)) *2. - 1.   #第一层
w2 = np.random.random(K+1)*2.-1.     #第二层

# logistic activation function
def logistic(x):
  return 1./(1+np.exp(-x))

# computes the network output for given inputs
def network(x):
  a = np.dot(w1,x)
  h = logistic(a)
  h[0] = 1.
  return np.dot(w2,h), h

# compute the loss for the whole dataset
def loss():
  Y = network(X)[0]
  loss = np.mean((Y-T)**2)
  return loss

# compute the gradient, i.e., for both w1 and w2
def gradient():
  # network output and hidden states for all inputs
  Y,H = network(X)

  # gradient for w2
  g2 = np.mean((Y-T) * H, axis=1)

  # gradient for w1
  g1 = (np.dot(np.outer(w2,Y-T) * H * (1-H),X.T))/len(X)

  #g1 = np.mean(np.dot(
  #    np.outer(w2,Y-T) * H * (1-H),
  #    X.T
  #))


  return g1, g2

# gradient descent
g1,g2 = gradient()

progression = []
for e in range(10000):
  # update weights
  w1 -= eta * g1
  w2 -= eta * g2

  # compute loss
  progression.append(loss())
  # add new gradient
  g1, g2 = gradient()

# plot everything together
x = np.arange(np.min(X),np.max(X),0.01)
x = np.vstack((np.ones(x.shape),x))
plt.plot(X[1],T,"rx-")
plt.plot(x[1],network(x)[0],"k-")
plt.savefig("Data.pdf")

# plt loss progression
plt.figure()
plt.plot(progression, label = "loss")
plt.legend()

plt.savefig("Loss.pdf")