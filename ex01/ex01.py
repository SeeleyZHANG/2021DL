import numpy
import random
import matplotlib
# 启用对latex 的支持
#matplotlib.rcParams["text.usetex"] = True
from matplotlib import pyplot

# create training data; assume x_0 = 1 in any case
# parameters are: mean per dimension, std per dimension, size,均值，方差，大小
# 从正态分布中抽出随机样本 normal(loc=0.0, scale=1.0, size=None)
neg = numpy.random.normal((1,-5,3),(0,2,2),(100,3))  #从正态分布里抽取随机样本
pos = numpy.random.normal((1,3,-5),(0,2,2),(100,3))

# collect in one list
samples = [(sample, -1) for sample in neg] + [(sample, 1) for sample in pos]

# y = w_0 * x_0 + w_1 * x_1 + w_2 * x_2

# initialize weights, 1*3的随机array
w = numpy.random.normal(size=3)

# stopping criterion: iterate untial all sample are classified correctly
# count number of incorrectly classified samples

# 初始化incorrect 当前错误的数量
incorrect = len(samples)

while incorrect > 0:
  # randomly shuffle list 随机打乱sample的顺序
  random.shuffle(samples)    #shuffle() 方法将序列的所有元素随机排序。先计算一共有多少个sample， 然后打乱其顺序

  # iterate over all samples
  incorrect = 0
  for x, t in samples:
    # predict class
    if numpy.dot(w, x) * t < 0:
      incorrect += 1
      w += t * x

# create figure in square shape
pyplot.figure(figsize=(6,6))

# plot points
pyplot.plot(neg[:,1], neg[:,2], "rx")
pyplot.plot(pos[:,1], pos[:,2], "gx")

# compute interaction from plane with z=0
# w_0 + w_1* x_1 + w_2 * x_2 = 0
# x_2 = (-w_0 - w_1 * x_1) / w_2
x_1 = numpy.array((-10,10))
x_2 = (-w[0] - w[1] *x_1) / w[2]

# plot line
pyplot.plot(x_1,x_2,"b-")

# finalize plot
pyplot.xlim(-10,10) # limit axes 
pyplot.ylim((-10,10))
pyplot.xlabel("$x_1$") # provide labels
pyplot.ylabel("$x_2$")
pyplot.show()

# write to file
pyplot.savefig("111.pdf")
