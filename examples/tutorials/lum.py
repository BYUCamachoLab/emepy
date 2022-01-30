# Import the needed libraries and modules
from emepy.lumerical import MSLumerical, MSLumerical1D
import numpy as np
from matplotlib import pyplot as plt


########### Finite Difference Solver ##############

# Create a modesolver object that represents a waveguide cross section
fd_solver = MSLumerical1D(
    1.55e-6,  # Set the wavelength of choice
    0.46e-6,  # Define the width of the waveguide
    0.22e-6,  # Define the thickness of the waveguide
    mesh=128,  # Set the mesh density
    num_modes=1,  # Set the number of modes to solve for
    eme_modes=False,
    PML=False
)

# Solve for the fundamental Eigenmode
fd_solver.solve()
fd_mode = fd_solver.get_mode()

# Plot the refractive index
plt.figure()
fd_mode.plot_material()
plt.show()

# Plot the eigenmode field components
plt.figure()
fd_mode.plot()
plt.show()

# Look at the effective index neff
print("FD Solver Effective index: ", fd_mode.get_neff())
