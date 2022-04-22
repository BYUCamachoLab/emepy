import numpy as np
from matplotlib import pyplot as plt

step = [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
fd_gradient = [41, -6.176, 0.7705, 0.169, 0.0111, 0.0082, 0.0118, 0.00822]

plt.figure()
plt.subplot(1,2,1)
plt.semilogx(step, np.abs(fd_gradient), marker='o')
plt.xlabel('Step Size')
plt.ylabel('Absolute Gradient')
plt.grid()
plt.subplot(1,2,2)
plt.loglog(step, np.abs(fd_gradient), marker='o', )
plt.xlabel('Step Size')
plt.ylabel('Absolute Gradient')
plt.grid()
plt.show()