import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

l1 = [1,2,3,4,5]
l2=[23,45,55,65,43]
plt.plot(l1,l2)
plt.show()
plt.savefig('test1.png')
