import matplotlib
#matplotlib.rcParams["text.usetex"] = True
from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

import scipy.linalg

# function to compute the loss for the given w1 and w2
# J_w = w_1 **2 + w_2 **2 + 30 * sin(w_1) * sin(w_2)
def compute_loss(w1, w2):
  return w1**2 + w2 ** 2+30. *np.sin(w1)*np.sin(w2)

# surface plot of the loss function
def plot_surface(alpha=.0):
  # define range of data samples
  w = np.arange(-10,10.001,0.1)
  w1, w2 = np.meshgrid(w,w)
  # compute the loss for all values of w1 and w2
  J = compute_loss(w1,w2) 
  # initialize 3d plot
  fig = pyplot.figure()
  # add_subplot切割画布
  ax = pyplot.figure().add_subplot(111,projection = "3d",azim = -40,elev = 50)
  # plot surface with jet colormap
  ax.plot_surface(w1,w2,J,cmap="jet",alpha=alpha)

  return fig,ax

# loss function 
def compute_gradient(w):
  return 2.*w+30.*np.cos(w) * np.sin(w[::-1])

# perform gradient descent from the given
def gradient_descent(w,eta):
  for j in range(1000):
    # stoppong criterion part 1: limit the number of iterations
    # compute gradient for current w
    g = compute_gradient(w)
    # stoppong criterion part 2: if norms of gradient is small, stop
    if scipy.linalg.norm(g) < 1e-4:
      break
    # perform one gradient descent stop
    w -= eta * g

    # return the optimised weight
    return w, j+1

# open pdf file
pdf = PdfPages("surface.pdf")

# start 10 trials with different initial weights
for trails in range(10):
  # create random weight in range (-10,10)
  w = np.random.random(2) * 20 - 10
  # perform gradient descent  
  o, iterations = gradient_descent(w.copy(),0.04)

# plot surface
fig, ax = plot_surface(.5)
# compute 2 values for initial and optimal weights
loss_w = compute_loss(w[0],w[1])
loss_o = compute_loss(o[0],o[1])
# plot values, connected with a line
ax.plot([w[0],o[0]],[w[1],o[1]],[loss_w,loss_o],"kx")
pdf.savefig(fig)

# print the number of iterations, the start and the final
print(iterations, w, o, loss_o)

# finalize and close pdf file
#pdf.close()