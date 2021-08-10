import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 18

import numpy
import scipy
import random

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages

# Training data
def input_data(which="iris"):
    import sklearn.datasets

    # get dataset and number of classes
    if which == "iris":
        dataset = sklearn.datasets.load_iris()
        o = 3
    else:
        dataset = sklearn.datasets.load_digits()
        o = 10
    # original data plus bias
    X = dataset.data
    X = numpy.hstack((numpy.ones((X.shape[0],1)),X))
    # target data as one-hot vector
    T = numpy.zeros((X.shape[0],o))
    for i,t in enumerate(dataset.target):
        T[i,t] = 1
    print(F"Loaded {len(X)} data samples")
    # provide input, one-hot targets and original target classes
    return X, T, dataset.target

# Network Implementation
def logistic(A):
    return 1./(1.+numpy.exp(-A))

def forward(X, W1, W2, output_z = True):
    # compute activation
    H = logistic(numpy.dot(W1,X))
    H[0,:] = 1 # bias

    # compute output
    Z = numpy.dot(W2, H)
    Y = numpy.exp(Z)
    Y /= numpy.sum(Y, axis=0)

    #return both
    if output_z:
        return Y,H,Z
    return Y, H


def loss(X, T, W1, W2):
    # compute output of network
    Y, H, Z = forward(X, W1, W2)

    # compute loss
    J = - numpy.sum(numpy.sum(T * Z, axis=0) - numpy.log(numpy.sum(numpy.exp(Z),axis=0)))
   
    # return everything
    return J, Y, H

# Learning algorithm
def gradient(X, T, Y, H, W1, W2):
    # first layer gradient
    G1 = numpy.dot(numpy.dot(W2.T, (Y-T)) * H * (1.-H), X.T)

    # second layer gradient
    G2 = numpy.dot((Y-T), H.T)

    # return both
    return G1, G2

old_update = None

def descent(X, T, W1, W2, eta, mu=None):
    # compute loss
    J, Y, H = loss(X, T, W1, W2)
    # compute gradient
    G1, G2 = gradient(X, T, Y, H, W1, W2)
    # update weights in-place
    W1 -= eta * G1
    W2 -= eta * G2

    if mu is not None:
        global old_update
        if old_update is not None:
            W1 += mu * old_update[0]
            W2 += mu * old_update[1]
        old_update = [-eta * G1, -eta * G2]

    # return the loss; weights are updated in-pleace
    return J


def batch(X, T, B, epochs):
    # get indexes list of all samples
    indexes = list(range(X.shape[0]))
    # start with empty batch
    batch = []
    end_of_epoch = False
    for epoch in range(epochs):
        # shuffle index before each epoch
        random.shuffle(indexes)
        # iterate over random samples
        for index in indexes:
            # append batch index
            batch.append(index)
            if len(batch) == B:
                # batch is full, yield the samples
                yield X[batch], T[batch], end_of_epoch
                # and clear batch
                batch.clear()
                end_of_epoch = False
        end_of_epoch = True
    # yield the last batch if not empty
    if batch:
        yield X[batch], T[batch], True



def accuracy(X, T, W1, W2):
    Y = forward(X.T, W1, W2, False)[0]

    return numpy.sum(numpy.argmax(Y,axis=0) == numpy.argmax(T.T, axis=0)) / len(T)

def stochastic_gradient_descent(X, T, W1, W2, batch_size=64, eta=0.001, mu=None, epochs=100000):
    print(F"Performing Stochastic Gradient Descent for {epochs} epochs with batch size {batch_size}")
    # perform gradient descent for the whole dataset
    losses = []
    # iterate over batches drawn from the data set
    for iteration, (x,t,end_of_epoch) in enumerate(batch(X, T, batch_size, epochs)):
        # perform one gradient descent step for the current batch
        J = descent(x.T, t.T, W1, W2, eta, mu)
        # compute classification accuracy
        A = accuracy(X, T, W1, W2)
        if end_of_epoch:
            losses.append(J)
        print("\riteration: ", iteration+1, "- Loss: ", J, end="", flush=True)
    print()
    return numpy.array(losses)

def command_line_options():
    # create command line parser object
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set some options with default values
    parser.add_argument("-d", "--database", default="iris", choices=("iris", "digits"), help="Select the database to be evaluated")
    parser.add_argument("-K", "--hidden", type=int, default=10, help="Select the number of hidden units")
    parser.add_argument("-e", "--epochs", type=int, default=100000, help="Select the number of epochs for GD")
    parser.add_argument("-B", "--batch-size", type=int, default=16, help="Select the batch size for SGD")
    parser.add_argument("-m", "--momentum", type=float, default=0.999, help="Select the momentum term")
    parser.add_argument("-l", "--learn-rate", type=float, default=5e-4, help="Set the learning rate")
    parser.add_argument("-s", "--seed", type=int, default=4, help="If selected, the given random seed is used")
    parser.add_argument("-o", "--plot", default="iris.pdf", help="Select the file where to write the plots into")

    # parse command line arguments
    return parser.parse_args()



if __name__ == '__main__':
    # get command line arguments
    args = command_line_options()

    if args.seed is not None:
        numpy.random.seed(args.seed)
    
    # read data
    X, T = X, T, C = input_data(args.dataset)

    # define number of hidden units
    K = args.hidden

    # initialize weights randomly
    W1 = numpy.random.random((K+1, X.shape[1])) * 2. - 1.
    W2 = numpy.random.random((T.shape[1], K+1)) * 2. - 1.
    
    sgd = stochastic_gradient_descent(X, T, W1, W2, args.batch_size, args.learn_rate, args.momentum, args.epochs)

    pdf = PdfPages(args.plot)

    # plot data points
    if args.dataset == "iris":
        # scatter plots for the original data
        pyplot.figure(figsize=(6,6))
        pyplot.scatter(X[C==0,1],X[C==0,2], c= "r", marker = "o", label="Setosa")
        pyplot.scatter(X[C==1,1],X[C==1,2], c= "g", marker = "o", label="Versicolor")
        pyplot.scatter(X[C==2,1],X[C==2,2], c= "b", marker = "o", label="Virginica")
        pyplot.axis("square")
        pyplot.xlabel("Sepal length ($x_1$")
        pyplot.ylabel("Sepal length ($x_2$")
        pyplot.xlim((4,8))
        pyplot.ylim((1.5,5))
        pyplot.legend(loc="upper right")
        pdf.savefig(bbox_inches='tight', pad_inches=0)

        pyplot.figure(figsize=(6,4))
        pyplot.scatter(X[C==0,3],X[C==0,4], c= "r", marker = "o", label="Setosa")
        pyplot.scatter(X[C==1,3],X[C==1,4], c= "g", marker = "o", label="Versicolor")
        pyplot.scatter(X[C==2,3],X[C==2,4], c= "b", marker = "o", label="Virginica")
        pyplot.axis("square")
        pyplot.xlabel("Sepal length ($x_3$")
        pyplot.ylabel("Sepal length ($x_4$")
        pyplot.xlim((0,8))
        pyplot.ylim((-.5,4))
        pyplot.legend(loc="upper left")
        pdf.savefig(bbox_inches='tight', pad_inches=0)
    
    # plot loss and accuracy progression into one plot
    pyplot.figure()
    # plot loss into first vertical axis (left)
    ax1 = pyplot.gca()
    ax1 = semilogx(losses[:,0],color="tab:blue")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("$\mathcal J^{\mathrm{CE}}$", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    # plot accuracy into second vertical axis
    ax2 = ax1.twinx()
    ax2 = semilogx(losses[:,1],color="tab:red")
    ax2.set_ylabel("Accuracy", color="tab:red")
    ax2.set_ylim((0,1))
    ax2.tick_params(axis='y', labelcolor="tab:red")
    # save figure
    pdf.savefig(bbox_inches='tight', pad_inches=0.1)

    pdf.close()