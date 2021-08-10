import random
import numpy as np
from numpy.core.fromnumeric import size
from matplotlib import pyplot

neg = np.random.normal((1,-5,3),(0,2,2),(100,3)) 
pos = np.random.normal((1,3,-5),(0,2,2),(100,3))

sample_neg = [(sample_neg, -1) for sample_neg in neg]
sample_pos = [(sample_pos, 1) for sample_pos in pos]
sample = sample_neg + sample_pos

w = np.random.normal(size = 3)

incorrect = len(sample)
while incorrect > 0:
    random.shuffle(sample)
    for x, t in sample:
        y = np.dot(w,x)
        if y * t < 0 :
            incorrect +=1
            w += t * x

# create figure in square shape
pyplot.figure(figsize=(6,6))

# plot points
pyplot.plot(neg[:,1], neg[:,2], "rx")
pyplot.plot(pos[:,1], pos[:,2], "gx")

pyplot.show()

x_1 = np.array((-10,10))
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
pyplot.savefig("p1.pdf")