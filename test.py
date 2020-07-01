import numpy as np
import matplotlib.pylab as plt

dat = np.random.randn(10,10)
plt.imshow(dat, interpolation='none')

clb = plt.colorbar()
clb.ax.set_title('This is a title')

plt.show()