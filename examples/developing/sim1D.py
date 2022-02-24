'''This script uses EMEPy's new 1D eigenmode solvers and 2D EME method

THIS IS UNDER DEVELOPMENT
'''

from emepy.fd import MSEMpy1D
import numpy as np
from matplotlib import pyplot as plt


# Simplest case, provide a width and cladding width and mesh
solver = MSEMpy1D(
    wl=1.55e-6,
    width=0.5e-6,
    num_modes=2,
    cladding_width=5e-6,
    mesh=128,
    accuracy=1e-8,
    boundary="0000",
)

plt.figure()
solver.plot_material()
plt.show()
solver.solve() 
for i in range(solver.num_modes):
    plt.figure()
    solver.get_mode(i).plot()
    plt.show()