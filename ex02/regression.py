import matplotlib.pyplot as pyplot
import numpy
import scipy.linalg # 线性代数库

# detect a certain line
a,b = 0.5, -2.0
# sample 100 x-locations in range (0,10)
X = numpy.random.random(100) * 10  # 返回随机的浮点数，在半开区间 [0.0, 1.0)。
# compute linear y-axis and add some noise
T = a*X + b + numpy.random.normal(0,a/2,X.shape)

# network output
# x can be a single sample, or serval samples
def y(X,w):
    y = w[0] + w[1] * X
    return y

# squarel loss function
def J(w):
    y = y(W,w)
    J = numpy.mean((y-T)**2)
    return numpy.mean((y(X,w)-T)**2)

# compute gradient at the given W
def gradient(w):
    # iterate over all samples and compute derivative w.r.t. w_0,w_1
    w_0 =  2 * numpy.mean(y(X,w)-T)
    w_1 = 2 * numpy.mean((y(X,w)-T)*X)
    g_w = numpy.array(w_0,w_1)
    return g_w 
    """return numpy.array((   # numpy.mean = 1/n
        2 * numpy.mean(y(X,w)-T),
        2 * numpy.mean((y(X,w)-T)*X)
    ))
"""
# select initial w vector; store to campare two methods later
initial_w = numpy.random.random(2)*2. - 1 # 初始化w 
w = initial_w.copy()

# selectlearning rate; larger values won't do
eta = 0.01
epochs = 0 # 初始化epoch

# compute first gradient
g = gradient(w) # 计算第一个gradient
# perform iterative gradient descent
# stopping critrtion : small norm of the gradient
while scipy.linalg.norm(g) > 1e-6:  #norm则表示范数
    # do one update step
    w -= eta * g
    # compute new gradient
    g = gradient(w)
    epochs+=1

# print number of epoches and final loss
print(epochs,J(w))

# plot sample, orignal line and optimal line
pyplot.plot(X,T,"k-")
pyplot.plot([0,10],[b,10*a+b],"r-")
pyplot.plot([0,10],[y(0,w),y(10,w)],"g-")
pyplot.legend(("Data","Source","Regresses"),loc="upper left")
pyplot.savefig("Linear.pdf")

### adaptive learning rate strategy
# take some initial weight as before
w = initial_w.copy()
epochs = 0

# compute gradient and loss
g = gradient(w)
old_j = J(w)

# perform iterative gradient descent
# with the same stopping criterion
s = scipy.linalg.norm(g)
while s > 1e-6:
    # do one update step
    w -= eta * g

    # compute updated loss
    j = J(w)

    # adapt learning rate
    if j >= old_j:
        eta += 0.5
    else:
        eta *= 1.1
    
    # compute new gradient and store current loss
    g = gradient(w)
    old_j = j
    epochs += 1

# print number of epochs and the final loss
print(epochs,J(w))