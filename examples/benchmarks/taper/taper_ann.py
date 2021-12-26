import emepy
from emepy.eme import Layer, EME
from emepy.mode import Mode
from emepy.ann import ANN, MSNeuralNetwork
import time

import numpy as np
import pylab

def taper_ann(print_s = True, start = 0, finish = 10):

    # This dictionary stores the information used by the benchmark scripts
    taper_ann_dict = {
        "density": [],
        "time": [],
        "s_params": []
    }

    # Cross sectional parameters (computational complexity determined here)
    ModeSolver = MSNeuralNetwork  # Choose a modesolver object that will calculate the 2D field profile
    num_modes = 1

    # Geometric parameters
    width1 = 0.8e-6  # Width of left waveguide
    thickness1 = 0.3e-6  # Thickness of left waveguide
    width2 = 0.25e-6  # Width of right waveguide
    thickness2 = 0.15e-6  # Thickness of right waveguide
    wavelength = 1.55e-6  # Wavelength of light (m)
    length = 10e-6  # Length of the waveguides
    taper_length = 2e-6  # The length of the taper

    wg_length = 0.5 * (length - taper_length)  # Length of each division in the taper

    ann = ANN()
    eme = EME()  # Choose either a normal eme or a periodic eme (PeriodicEME())

    for taper_density in range(start, finish):

        # Ensure a new eme each iteration
        eme.reset()

        # first layer is a straight waveguide
        mode1 = ModeSolver(
            ann,
            wavelength,
            width1,
            thickness1,
        )
        straight1 = Layer(mode1, num_modes, wavelength, wg_length)
        eme.add_layer(straight1)

        # create the discrete taper with a fine enough taper density to approximate a continuous linear taper
        widths = np.linspace(width1, width2, taper_density)
        thicknesses = np.linspace(thickness1, thickness2, taper_density)
        taper_length_per = taper_length / taper_density if taper_density else None

        # add the taper layers
        for i in range(taper_density):
            solver = ModeSolver(ann, wavelength, widths[i], thicknesses[i])
            taper_layer = Layer(solver, num_modes, wavelength, taper_length_per)
            eme.add_layer(taper_layer)

        # last layer is a straight waveguide of smaller geometry
        mode2 = ModeSolver(ann, wavelength, width2, thickness2)
        straight2 = Layer(mode2, num_modes, wavelength, wg_length)
        eme.add_layer(straight2)

        # eme.draw()  # Look at our simulation geometry

        t1 = time.time()
        eme.propagate()  # Run the eme
        t2 = time.time()
        taper_ann_dict["time"].append(t2-t1)
        taper_ann_dict["density"].append(taper_density)
        taper_ann_dict["s_params"].append(eme.s_parameters())

        if print_s:
            print(taper_density,": ",np.abs(eme.s_parameters()))  # Extract s_parameters

    return taper_ann_dict

def main():

    taper_ann()

if __name__ == "__main__":
    main()


