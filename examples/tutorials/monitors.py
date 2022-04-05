# # from emepy.ann import ANN, MSNeuralNetwork
# from emepy.fd import MSEMpy
# from emepy.eme import Layer, EME
# import numpy as np
# from matplotlib import pyplot as plt

# num_periods = 3  # Number of Periods for Bragg Grating
# length = .159 # Length of each segment of BG, Period = Length * 2
# wavelength = 1.55 # Wavelength
# num_modes = 1  # Number of Modes
# mesh = 128 # Number of mesh points
# width1 = 0.46 # Width of first core block
# width2 = 0.54 # Width of second core block 
# thickness = 0.22 # Thicnkess of the core
# modesolver = MSEMpy # Which modesolver to use

# eme = EME(num_periods=num_periods)
# # ann = ANN()

# mode_solver1 = modesolver(
#     # ann,
#     wavelength,
#     width1,
#     thickness,
# )  # First half of bragg grating

# mode_solver2 = modesolver(
#     # ann,
#     wavelength,
#     width2,
#     thickness,
# )  # Second half of bragg grating

# eme.add_layer(Layer(mode_solver1, num_modes, wavelength, length))  # First half of bragg grating
# eme.add_layer(Layer(mode_solver2, num_modes, wavelength, length))  # Second half of bragg grating

# eme.reset(full_reset=False)

# monitor = eme.add_monitor(axes="xz")

# eme.propagate()  # propagate at given wavelength

# # plt.figure()
# # monitor.visualize(component="n")
# # plt.colorbar()
# # plt.show()

# # plt.figure()
# # monitor.visualize(component="Hy")
# # plt.colorbar()
# # plt.show()

# eme.reset(full_reset=False)

# monitor = eme.add_monitor(axes="yz")

# eme.propagate()  # propagate at given wavelength

# plt.figure()
# monitor.visualize(component="n")
# plt.colorbar()
# plt.show()

# plt.figure()
# monitor.visualize(component="Hy")
# plt.colorbar()
# plt.show()

import emepy
from emepy import Layer, EME, Mode, MSEMpy
import numpy as np
from matplotlib import pyplot as plt
from emepy.tools import Si, SiO2

# Geometric parameters
wavelength = 1.55 # Wavelength
width = 0.4  # Width of left waveguide
gap = 0.2 # Gap between waveguides
thickness = 0.22  # Thickness of left waveguide
num_modes=2 # Number of modes
mesh=100 # Number of mesh points
core_index=Si(wavelength) # Silicon core
cladding_index=SiO2(wavelength) # Oxide cladding
x = np.linspace(-2,2,mesh)
n = np.ones(mesh) * cladding_index

# Create simulation 
eme_2 = EME()

# Create left waveguide
single_left_edge = -gap/2-width
single_right_edge = -gap/2
single_n = np.where((single_left_edge <= x) * (x <= single_right_edge), core_index, n)

single_channel = MSEMpy(
    wavelength,
    width=None,
    thickness=thickness,
    cladding_index=cladding_index,
    num_modes=num_modes,
    mesh=mesh,
    x=x,
    y=x,
    n=single_n
)

# Create left waveguide
left_edge = -gap/2-width
right_edge = -gap/2
n = np.where((left_edge <= x) * (x <= right_edge), core_index, n)

# Create right waveguide
left_edge = gap/2
right_edge = gap/2+width
n = np.where((left_edge <= x) * (x <= right_edge), core_index, n)

two_channel = MSEMpy(
    wavelength,
    width=None,
    thickness=thickness,
    cladding_index=cladding_index,
    num_modes=num_modes,
    mesh=mesh,
    x=x,
    y=x,
    n=n
)

eme_2.add_layer(Layer(single_channel, num_modes, wavelength, 0.5))  # First half of bragg grating
eme_2.add_layer(Layer(two_channel, num_modes, wavelength, 25))  # Second half of bragg grating

monitor = eme_2.add_monitor(axes="xyz")

eme_2.propagate()  # propagate at given wavelength

plt.figure()
monitor.visualize(component="n", axes="xz")
plt.colorbar()
plt.show()

plt.figure()
monitor.visualize(component="Hy", axes="xz")
plt.colorbar()
plt.show()