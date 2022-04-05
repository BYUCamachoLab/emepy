import numpy as np
from matplotlib import pyplot as plt

mesh = np.linspace(50, 200, 6)
adjoint = [-3.764279174808804e-06, -3.2254942237182947e-06, 1.8415695609303942e-05, 9.972245562733456e-06, -2.2152437591608423e-05, 2.8012812884963418e-05]
fd = [-0.0015863637103439299, 4.129771794103565e-05, 6.886275593864788e-06, 5.933889876130749e-07, -7.087378481873685e-07, -1.10707086442563e-06]

plt.figure()
plt.plot(mesh[1:], adjoint[1:], label="adjoint")
plt.plot(mesh[1:], fd[1:], label="finite difference")
plt.legend()
plt.show()