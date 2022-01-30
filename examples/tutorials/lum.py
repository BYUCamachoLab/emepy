# # Import the needed libraries and modules
# from emepy.lumerical import MSLumerical, MSLumerical1D, LumEME
# from emepy.eme import Layer
# import numpy as np
# from matplotlib import pyplot as plt


# ########### Finite Difference Solver ##############

# # Create a modesolver object that represents a waveguide cross section
# eme = LumEME()

# fd_solver1 = MSLumerical1D(
#     1.55e-6,  # Set the wavelength of choice
#     0.46e-6,  # Define the width of the waveguide
#     0.22e-6,  # Define the thickness of the waveguide
#     mesh=128,  # Set the mesh density
#     num_modes=5,  # Set the number of modes to solve for
#     eme_modes=False,
#     PML=False,
#     mode=eme.mode
# )
# eme.add_layer(Layer(fd_solver1,5,1.55e-6, 0.5e-6))

# fd_solver2 = MSLumerical1D(
#     1.55e-6,  # Set the wavelength of choice
#     0.90e-6,  # Define the width of the waveguide
#     0.22e-6,  # Define the thickness of the waveguide
#     mesh=128,  # Set the mesh density
#     num_modes=5,  # Set the number of modes to solve for
#     eme_modes=False,
#     PML=False,
#     mode=eme.mode
# )
# eme.add_layer(Layer(fd_solver2,5,1.55e-6, 0.5e-6))

# monitor = eme.add_monitor(axes="xz")

# eme.propagate()

# plt.figure()
# monitor.visualize(axes="xz", component="Hx")
# plt.show()


# # Solve for the fundamental Eigenmode
# fd_solver.solve()
# fd_mode = fd_solver.get_mode()

# # Plot the refractive index
# plt.figure()
# fd_mode.plot_material()
# plt.show()

# # Plot the eigenmode field components
# plt.figure()
# fd_mode.plot()
# plt.show()

# # Look at the effective index neff
# print("FD Solver Effective index: ", fd_mode.get_neff())








from emepy.lumerical import MSLumerical, MSLumerical1D, LumEME
from emepy.eme import Layer
import emepy
import numpy as np
from matplotlib import pyplot as plt

# Design parameters
taper_length = 7e-6  # The length of the taper
taper_density = 5 # Number of taper segments
alpha = 3 # Strength of function (either tanh or bezier) -> 0 = linear
type_tanh = True # "bezier"

def taper_func(start, end, num_points):

    # Linear Curve
    x = np.linspace(width1, width2, taper_density)

    # Tanh Curve
    if type_tanh:
        xt = x - np.min(x)
        xta = xt / np.max(xt)
        tanh = np.tanh(alpha*(xta-0.5))+1
        tanh -= np.min(tanh)
        tanh *= np.max(xt) / np.max(tanh)
        tanh += np.min(x)
        return tanh
    # Bezier Curve
    else:
        return None

# Geometric parameters
width1 = 0.5e-6  # Width of left waveguide
thickness1 = 0.22e-6  # Thickness of left waveguide
width2 = 7e-6  # Width of right waveguide
thickness2 = 0.22e-6  # Thickness of right waveguide
wavelength = 1.55e-6  # Wavelength of light (m)
length = 3e-6  # Length of the waveguides
num_modes_first_half = 10 # Number of modes to solve for
num_modes_second_half = 20 # Number of modes to solve for
mesh=128 # Number of mesh points in each xy dimension

eme = LumEME()  # Choose either a normal eme or a periodic eme (PeriodicEME())

# first layer is a straight waveguide
mode1 = MSLumerical1D(
    wavelength,
    width1,
    thickness1,
    num_modes=3,
    cladding_width=10e-6,
    cladding_thickness=10e-6,
    mesh=mesh,
    mode=eme.mode
)
straight1 = Layer(mode1, 3, wavelength, length)
eme.add_layer(straight1)

# create the discrete taper with a fine enough taper density to approximate a continuous linear taper
widths = taper_func(width1, width2, taper_density)
thicknesses = np.linspace(thickness1, thickness2, taper_density)
taper_length_per = taper_length / taper_density

# add the taper layers
for i in range(taper_density):
    num_modes = num_modes_first_half if i < taper_density / 2.0 else num_modes_second_half
    solver = MSLumerical1D(wavelength, widths[i], thicknesses[i], num_modes=num_modes, mesh=mesh,cladding_width=10e-6,cladding_thickness=10e-6,mode=eme.mode)
    taper_layer = Layer(solver, num_modes, wavelength, taper_length_per)
    eme.add_layer(taper_layer)

# last layer is a straight waveguide of smaller geometry
mode2 = MSLumerical1D(wavelength, width2, thickness2, num_modes=num_modes_second_half, mesh=mesh,cladding_width=10e-6,cladding_thickness=10e-6,mode=eme.mode)
straight2 = Layer(mode2, num_modes_second_half, wavelength, length)
eme.add_layer(straight2)

monitor = eme.add_monitor(axes="xz")

# eme.draw() 
# plt.show()

eme.propagate()  # Run the eme

plt.figure()
monitor.visualize(component="Ex")
plt.colorbar()
plt.show()