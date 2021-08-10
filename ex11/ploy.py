import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.size"] = 18
from matplotlib import  pyplot
from matplotlib.backends.backend_pdf import  PdfPages

import csv
import numpy

# read, split and convert fuls to float
results = numpy.array([
                       [
                        [
                         float(v) for v in line
                        ]
                        for line in csv.reader(open(f"Results_{fn}.txt"),delimiter=",")
                       ]
                       for fn in ("none","noise","adv")
])

# plot everything into one multi-page pdf
pdf = PdfPages("Adversarial.pdf")

pyplot.figure()
# three lines for clean accuracy
pyplot.plot(results[0.0],"r-",label="None")
pyplot.plot(results[1,0],"b-",label="Noise")
pyplot.plot(results[2,0],"g-",label="FGS")

# legend and axis labels
pyplot.legend()
pyplot.xlabel("Epoch")
pyplot.ylabel("Accuracy (clean)")

# save figure to pdf
pdf.savefig(bbox_inches='tight',pad_inches=0)

# assure that odf is saved
pdf.close()