import matplotlib
#matplotlib.rcParams["text.usetex"] = True
#matplotlib.rcParams["font.size"] = 18

import numpy
import csv
import random

from matplotlib import pyplot

def load_data(filename,skip_header=False,normalize_data=False):
    with open(filename) as f:
        r = csv.reader(f,delimiter=",")
        # skip header line
        if skip_header:
            next(r)
        X, T = [], []
        for splits in r:
            X.append([float(s) if s != "NA" else -1. for s in splits[:-1]])
            T.append(float(splits[-1]))

        # normalize data to range (0,1)
        if normalize_data:
            X /= numpy.max(X, axis=0)
        # add bias neuron
        X = numpy.hstack((numpy.ones((len(X),1)), numpy.array(X)))
        T = numpy.array(T)
        print(F"Loaded Datase with {len(X)} samples of {X.shape[1]} dimensions and bias {1.-numpy.sum(T) / len(T)}")
        return X, T

def logistic(A):
    return 1./(1.+numpy.exp(-A))

def forward(X, W1, W2, output_z = False):
    # compute activation
    H = logistic(numpy.dot(W1,X))
    H[0,:] = 1 # bias

    # compute output
    Z = numpy.dot(W2, H)
    Y = logistic(Z)
    # Y=Z

    #return both
    if output_z:
        return Y,H,Z
    return Y, H


def loss(X, T, W1, W2):
    # compute output of network
    Y, H = forward(X, W1, W2)

    # compute loss
    # J = - numpy.sum(T * numpy.log(Y) + (1-T) * numpy.log(1-Y).T)
    J = - numpy.dot(T, numpy.log(Y).T - numpy.dot((1-T), numpy.log(1-Y).T))

    # return everything
    return float(J), Y, H

def gradient(X, T, Y, H, W1, W2):
    # first layer gradient
    G1 = numpy.dot(numpy.dot(W2.T, (Y-T)) * H * (1.-H), X.T)

    # second layer gradient
    G2 = numpy.dot((Y-T), H.T)

    # return both
    return G1, G2

# old_update = None

def descent(X, T, W1, W2, eta):
    # compute loss
    J, Y, H = loss(X, T, W1, W2)
    # compute gradient
    G1, G2 = gradient(X, T, Y, H, W1, W2)
    # update weights in-place
    W1 -= eta * G1
    W2 -= eta * G2

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
    P = Y > 0.5
    return numpy.sum(P == T) / len(T)

def stochastic_gradient_descent(X, T, W1, W2, batch_size=64, eta=0.001, mu=None, epochs=100000):
    print(F"Performing Stochastic Gradient Descent for {epochs} epochs with batch size {batch_size}")
    # perform gradient descent for the whole dataset
    losses = []
    try:
        # iterate over batches drawn from the data set
        for iteration, (x,y,e) in enumerate(batch(X, T, batch_size, epochs)):
            # perform one gradient descent step for the current batch
            J = descent(x.T,T.T, W1, W2, eta)
            A = accuracy(X, T, W1, W2)
            if e:
                losses.append(J)
            print("\riteration: ", iteration+1, "- Loss: ", J, end="", flush=True)
    except KeyboardInterrupt:
        pass
    print()
    return numpy.array(losses)
def command_line_options():
    # create command line parser object
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # set some options with default values
    parser.add_argument("-d", "--database", default="/Users/xinyi/Desktop/2021Spring/Deep Learning/code/spambase/spambase.data", help="Select the database to use for training")
    parser.add_argument("-x", "--skip-header", action="store_true", help="Shall the first line of the dataset be skipped?")
    parser.add_argument("-n", "--normalize_data", action="store_true", help="Shall the input be normalized?")
    parser.add_argument("-K", "--hidden", type=int, default=10, help="Select the number of hidden units")
    parser.add_argument("-e", "--epochs", type=int, default=1000, help="Select the number of epochs for GD")
    parser.add_argument("-B", "--batch-size", type=int, default=64, help="Select the batch size for SGD")
    parser.add_argument("-l", "--learn-rate", type=float, default=1e-4, help="Set the learning rate")
    parser.add_argument("-s", "--seed", type=int, default=4, help="If selected, the given random seed is used")
    parser.add_argument("-o", "--plot", default="spambase.pdf", help="Select the file where to write the plots into")

    # parse command line arguments
    return parser.parse_args()


if __name__ == '__main__':
    # get command line arguments
    args = command_line_options()

    if args.seed is not None:
        numpy.random.seed(args.seed)
    
    # read data
    X, T = load_data(args.database, args.skip_header, args.normalize_data)

    # define number of hidden units
    K = args.hidden

    # initialize weights randomly
    W1 = numpy.random.random((K+1, X.shape[1])) * 2. - 1.
    W2 = numpy.random.random((1, K+1)) * 2. - 1.
    
    losses = stochastic_gradient_descent(X, T, W1, W2, args.batch_size, args.learn_rate, args.epochs)

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

    pyplot.savefig(args.plot)