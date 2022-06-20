from emepy.lumerical import MSLumerical, LumEME  # Requires Lumerical API
from emepy.eme import Layer
import time

import numpy as np


def taper_lumerical(print_s=True, start=0, finish=10):

    # This dictionary stores the information used by the benchmark scripts
    taper_lumerical_dict = {"density": [], "time": [], "s_params": []}

    # Cross sectional parameters (computational complexity determined here)
    ModeSolver = MSLumerical  # Choose a modesolver object that will calculate the 2D field profile
    mesh = 128  # Mesh density of 2D field profiles
    num_modes = 1

    # Geometric parameters
    width1 = 0.8  # Width of left waveguide
    thickness1 = 0.3  # Thickness of left waveguide
    width2 = 0.25  # Width of right waveguide
    thickness2 = 0.15  # Thickness of right waveguide
    wavelength = 1.55  # Wavelength of light (m)
    length = 10  # Length of the waveguides
    # taper_density = 10  # How many divisions in the taper where eigenmodes will be calculated
    taper_length = 2  # The length of the taper

    wg_length = 0.5 * (length - taper_length)  # Length of each division in the taper

    eme = LumEME()  # Choose either a normal eme or a periodic eme (PeriodicEME())
    for taper_density in range(start, finish):

        # Ensure a new eme each iteration
        eme.reset()

        # first layer is a straight waveguide
        mode1 = ModeSolver(
            wl=wavelength, width=width1, thickness=thickness1, mesh=mesh, num_modes=num_modes, mode=eme.mode
        )
        straight1 = Layer(mode1, num_modes, wavelength, wg_length)
        eme.add_layer(straight1)

        # create the discrete taper with a fine enough taper density to approximate a continuous linear taper
        widths = np.linspace(width1, width2, taper_density)
        thicknesses = np.linspace(thickness1, thickness2, taper_density)
        taper_length_per = taper_length / taper_density if taper_density else None

        # add the taper layers
        for i in range(taper_density):
            solver = ModeSolver(
                wl=wavelength, width=widths[i], thickness=thicknesses[i], mesh=mesh, num_modes=num_modes, mode=eme.mode
            )
            taper_layer = Layer(solver, num_modes, wavelength, taper_length_per)
            eme.add_layer(taper_layer)

        # last layer is a straight waveguide of smaller geometry
        mode2 = ModeSolver(
            wl=wavelength, width=width2, thickness=thickness2, mesh=mesh, num_modes=num_modes, mode=eme.mode
        )
        straight2 = Layer(mode2, num_modes, wavelength, wg_length)
        eme.add_layer(straight2)

        # eme.draw()  # Look at our simulation geometry

        t1 = time.time()
        eme.propagate()  # Run the eme
        t2 = time.time()
        taper_lumerical_dict["time"].append(t2 - t1)
        taper_lumerical_dict["density"].append(taper_density)
        taper_lumerical_dict["s_params"].append(eme.s_parameters())

        if print_s:
            print(taper_density, ": ", np.abs(eme.s_parameters()))  # Extract s_parameters

    return taper_lumerical_dict


def main():

    taper_lumerical()


if __name__ == "__main__":

    main()
