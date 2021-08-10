import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 18
from matplotlib import pyplot, patches
from matplotlib.backends.backend_pdf import PdfPages

import torch
import numpy

from rbf import network, test_loader

# extract test set
with torch.no_grad():
    X,T = [],[]
    for (x,t) in test_loader:
        X.append(x)
        T.append(t)
    X = torch.cat(X)
    fc1 = network.extract(X).numpy()
    T = torch.cat(T).numpy()

# centers are the RBF W-matrix
centers = network.rbf.W.detach().numpy()
# variances from the RBF layer
variances = network.rbf.var.detach(),numpy()

# 10 different colors for plotting
colors = numpy.array([
                      [230,25,75],
                      [60, 180, 75],
                      [255, 255, 25],
                      [67, 99, 216],
                      [245, 130, 49],
                      [145, 130, 49],
                      [70, 240, 240],
                      [240, 50, 230],
                      [188, 246, 12],
                      [250, 190, 190],
]) / 255.

# scatter plot
pyplot.figure(figsize=(6,5))
pyplot.scatter(
    fc1[:,0],
    fc1[:,1],
    c=colors[T],
    edgecolors='none',
    s=2
)

# labels
pyplot.xlabel("$\\varphi_1$")
pyplot.ylabel("$\\varphi_2$")

pyplot.savefig("Scatter.pdf", bbox_inches='tight',pad_inches=0)

# learned centers
pyplot.scatter(
    centers[:,0],
    centers[:,1],
    c="k",
    marker="+",
    s=variances * 40
)

pyplot.savefig("Scatter+.pdf", bbox_inches="tight", pad_inches=0)