import emepy
from emepy.FD_modesolvers import ModeSolver_Lumerical  # Requires Lumerical API
from emepy.FD_modesolvers import ModeSolver_EMpy  # Open source
from emepy.eme import Layer, EMERunner
from emepy.mode import Mode

import numpy as np
import pylab


# Cross sectional parameters (computational complexity determined here)
ModeSolver = ModeSolver_EMpy  # Choose a modesolver object that will calculate the 2D field profile
mesh = 256  # Mesh density of 2D field profiles
num_modes = 2

# Geometric parameters
width1 = 0.6e-6  # Width of left waveguide
thickness1 = 0.4e-6  # Thickness of left waveguide
width2 = 0.5e-6  # Width of right waveguide
thickness2 = 0.3e-6  # Thickness of right waveguide
wavelength = 1.55e-6  # Wavelength of light (m)
length = 10e-6  # Length of the waveguides
taper_density = 10  # How many divisions in the taper where eigenmodes will be calculated
taper_length = 2e-6  # The length of the taper

wg_length = 0.5 * (length - taper_length)  # Length of each division in the taper

eme = EMERunner()  # Choose either a normal eme or a periodic eme (PeriodicEME())

# first layer is a straight waveguide
mode1 = ModeSolver(
    wl=wavelength,
    width=width1,
    thickness=thickness1,
    mesh=mesh,
    num_modes=num_modes,
    # lumapi_location="/Applications/Lumerical v202.app/Contents/API/Python",
)
straight1 = Layer(mode1, num_modes, wavelength, wg_length)
eme.add_layer(straight1)

# create the discrete taper with a fine enough taper density to approximate a continuous linear taper
widths = np.linspace(width1, width2, taper_density)
thicknesses = np.linspace(thickness1, thickness2, taper_density)
taper_length_per = taper_length / taper_density if taper_density else None

# add the taper layers
for i in range(taper_density):
    solver = ModeSolver(wl=wavelength, width=widths[i], thickness=thicknesses[i], mesh=mesh, num_modes=num_modes)
    taper_layer = Layer(solver, num_modes, wavelength, taper_length_per)
    eme.add_layer(taper_layer)

# last layer is a straight waveguide of smaller geometry
mode2 = ModeSolver(wl=wavelength, width=width2, thickness=thickness2, mesh=mesh, num_modes=num_modes)
straight2 = Layer(mode2, num_modes, wavelength, wg_length)
eme.add_layer(straight2)

# eme.draw()  # Look at our simulation geometry

eme.propagate()  # Run the eme

print(np.abs(eme.get_s_params()))  # Extract s_parameters
