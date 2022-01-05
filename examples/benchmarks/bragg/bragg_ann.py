from emepy.ann import ANN, MSNeuralNetwork
from emepy.eme import Layer, EME
import numpy as np
from matplotlib import pyplot as plt
import argparse


def bragg_ann(args):

    num_periods = args.num_periods  # Number of Periods for Bragg Grating
    length = args.length  # Length of each segment of BG, Period = Length * 2
    num_wavelengths = args.num_wavelengths  # Number of wavelengths to sweep
    wl_lower = args.wl_lower  # Lower wavelength bound
    wl_upper = args.wl_upper  # Upper wavelength bound
    num_modes = args.num_modes  # Number of Modes
    mesh = args.mesh  # Number of mesh points
    width1 = args.width1  # Width of first core block
    width2 = args.width2  # Width of second core block
    thickness = args.thickness  # Thicnkess of the core
    modesolver = MSNeuralNetwork  # Which modesolver to use
    t = []  # Array that holds the transmission coefficients for different wavelengths

    eme = EME(num_periods=num_periods)
    ann = ANN()

    for wavelength in np.linspace(wl_lower, wl_upper, num_wavelengths):

        eme.reset()

        mode_solver1 = modesolver(
            ann,
            wavelength * 1e-6,
            width1 * 1e-6,
            thickness * 1e-6,
        )  # First half of bragg grating

        mode_solver2 = modesolver(
            ann,
            wavelength * 1e-6,
            width2 * 1e-6,
            thickness * 1e-6,
        )  # Second half of bragg grating

        eme.add_layer(
            Layer(mode_solver1, num_modes, wavelength * 1e-6, length * 1e-6)
        )  # First half of bragg grating
        eme.add_layer(
            Layer(mode_solver2, num_modes, wavelength * 1e-6, length * 1e-6)
        )  # Second half of bragg grating

        eme.propagate()  # propagate at given wavelength

        t.append(
            np.abs((eme.s_parameters()))[0, 0, 1] ** 2
        )  # Grab the transmission coefficient

    return t


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-num_periods",
        type=int,
        default="50",
        help="Number of Periods for Bragg Grating (default: 50)",
    )
    parser.add_argument(
        "-length",
        type=float,
        default="0.159",
        help="Length of each segment of BG, Period = Length * 2 (microns) (default: 0.159)",
    )
    parser.add_argument(
        "-num_wavelengths",
        type=int,
        default="30",
        help="Number of wavelengths to sweep (default: 30)",
    )
    parser.add_argument(
        "-wl_lower",
        type=float,
        default="1.5",
        help="Lower wavelength bound (microns) (default: 1.5)",
    )
    parser.add_argument(
        "-wl_upper",
        type=float,
        default="1.6",
        help="Upper wavelength bound (microns) (default: 1.6)",
    )
    parser.add_argument(
        "-num_modes", type=int, default="1", help="Number of Modes (default: 1)"
    )
    parser.add_argument(
        "-mesh", type=int, default="128", help="Number of mesh points (default: 128)"
    )
    parser.add_argument(
        "-width1",
        type=float,
        default="0.46",
        help="Width of first core block (microns) (default: 0.46)",
    )
    parser.add_argument(
        "-width2",
        type=float,
        default="0.54",
        help="Width of second core block  (microns) (default: 0.54)",
    )
    parser.add_argument(
        "-thickness",
        type=float,
        default="0.22",
        help="Thicnkess of the core (microns) (default: 0.22)",
    )

    args = parser.parse_args()
    t = bragg_ann(args)

    # Plot the results
    plt.plot(np.linspace(args.wl_lower, args.wl_upper, args.num_wavelengths), t)
    plt.title("Grating freq sweep nperiods=" + str(args.num_periods))
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("t")
    plt.grid()
    plt.show()


if __name__ == "__main__":

    main()
