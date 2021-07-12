import emepy
from emepy.FD_modesolvers import ModeSolver_Lumerical  # Requires Lumerical API
from emepy.FD_modesolvers import ModeSolver_EMpy  # Open source
from emepy.eme import Layer, EME
from emepy.mode import Mode

import numpy as np
import pylab


# Cross sectional parameters (computational complexity determined here)
ModeSolver = ModeSolver_Lumerical  # Choose a modesolver object that will calculate the 2D field profile
mesh = 500  # Mesh density of 2D field profiles
num_modes = 1

# Geometric parameters
width1 = 0.8e-6  # Width of left waveguide
thickness1 = 0.3e-6  # Thickness of left waveguide
width2 = 0.25e-6  # Width of right waveguide
thickness2 = 0.15e-6  # Thickness of right waveguide
wavelength = 1.55e-6  # Wavelength of light (m)
length = 10e-6  # Length of the waveguides
# taper_density = 10  # How many divisions in the taper where eigenmodes will be calculated
taper_length = 2e-6  # The length of the taper

wg_length = 0.5 * (length - taper_length)  # Length of each division in the taper

eme = EME()  # Choose either a normal eme or a periodic eme (PeriodicEME())

for taper_density in range(21, 51):

    # Ensure a new eme each iteration
    eme.reset()

    # first layer is a straight waveguide
    mode1 = ModeSolver(
        wl=wavelength,
        width=width1,
        thickness=thickness1,
        mesh=mesh,
        num_modes=num_modes,
        lumapi_location="/Applications/Lumerical\ v202.app/Contents/API/Python/",
    )
    straight1 = Layer(mode1, num_modes, wavelength, wg_length)
    eme.add_layer(straight1)

    # eme.draw()

    # create the discrete taper with a fine enough taper density to approximate a continuous linear taper
    widths = np.linspace(width1, width2, taper_density)
    thicknesses = np.linspace(thickness1, thickness2, taper_density)
    taper_length_per = taper_length / taper_density if taper_density else None

    # add the taper layers
    for i in range(taper_density):
        solver = ModeSolver(wl=wavelength, width=widths[i], thickness=thicknesses[i], mesh=mesh, num_modes=num_modes)
        taper_layer = Layer(solver, num_modes, wavelength, taper_length_per)
        eme.add_layer(taper_layer)

    # eme.draw()

    # last layer is a straight waveguide of smaller geometry
    mode2 = ModeSolver(wl=wavelength, width=width2, thickness=thickness2, mesh=mesh, num_modes=num_modes)
    straight2 = Layer(mode2, num_modes, wavelength, wg_length)
    eme.add_layer(straight2)

    # eme.draw()

    # eme.draw()  # Look at our simulation geometry

    eme.propagate()  # Run the eme

    print(taper_density,": ",np.abs(eme.get_s_params()))  # Extract s_parameters



# 0 :  [[[0.24169941 0.74486248]
#   [0.74486248 0.24169941]]]
# 1 :  [[[0.24169941 0.74486248]
#   [0.74486248 0.24169941]]]
# 2 :  [[[0.24169941 0.74486248]
#   [0.74486248 0.24169941]]]
# 3 :  [[[0.14792082 0.70015021]
#   [0.70015021 0.14023991]]]
# 4 :  [[[0.05938674 0.72624987]
#   [0.72624987 0.06335213]]]
# 5 :  [[[0.04621868 0.76109881]
#   [0.76109881 0.04404517]]]
# 6 :  [[[0.04329289 0.79368749]
#   [0.79368749 0.03693121]]]
# 7 :  [[[0.00821884 0.82142849]
#   [0.82142849 0.00554595]]]
# 8 :  [[[0.01899095 0.84254567]
#   [0.84254567 0.0169274 ]]]
# 9 :  [[[0.01308279 0.85958923]
#   [0.85958923 0.01247748]]]
# 10 :  [[[0.009549   0.87329611]
#   [0.87329611 0.00944152]]]
# (base) tmpfac-126-7:examples ianhammond$ python3 taper.py
# 11 :  [[[0.00767933 0.88457234]
#   [0.88457234 0.00772951]]]
# 12 :  [[[0.00649612 0.89408222]
#   [0.89408222 0.00651798]]]
# 13 :  [[[0.00585257 0.90212412]
#   [0.90212412 0.00589219]]]
# 14 :  [[[0.00501536 0.90914502]
#   [0.90914502 0.00508077]]]
# 15 :  [[[0.00460919 0.91527816]
#   [0.91527816 0.00464512]]]
# 16 :  [[[0.00461373 0.9204365 ]
#   [0.9204365  0.00466529]]]
# 17 :  [[[0.00441459 0.92518947]
#   [0.92518947 0.00442748]]]
# 18 :  [[[0.00410909 0.92941657]
#   [0.92941657 0.00411845]]]
# 19 :  [[[0.00409033 0.93310102]
#   [0.93310102 0.00409809]]]
# 20 :  [[[0.00414627 0.9364613 ]
#   [0.9364613  0.00416259]]]
# (base) tmpfac-126-7:examples ianhammond$ python3 taper.py
# 21 :  [[[0.00426469 0.93945414]
#   [0.93945414 0.00427932]]]
# 22 :  [[[0.0039222  0.94220403]
#   [0.94220403 0.00395028]]]
# 23 :  [[[0.00377989 0.94474017]
#   [0.94474017 0.0038158 ]]]
# 24 :  [[[0.00386355 0.9470418 ]
#   [0.9470418  0.00388628]]]
# 25 :  [[[0.00402305 0.94917219]
#   [0.94917219 0.00403223]]]
# 26 :  [[[0.00442391 0.95109694]
#   [0.95109694 0.00438393]]]
# 27 :  [[[0.00385554 0.95296334]
#   [0.95296334 0.00386894]]]
# 28 :  [[[0.00385715 0.95460753]
#   [0.95460753 0.00386167]]]
# 29 :  [[[0.00384701 0.95633344]
#   [0.95633344 0.00384637]]]
# 30 :  [[[0.00405014 0.95768274]
#   [0.95768274 0.00405136]]]
# 31 :  [[[0.00429439 0.95905079]
#   [0.95905079 0.00428134]]]