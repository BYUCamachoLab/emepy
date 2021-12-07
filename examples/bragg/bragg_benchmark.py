from matplotlib import pyplot as plt
import numpy as np
import argparse
from bragg_ann import bragg_ann
from bragg_lumerical import bragg_lumerical
from bragg_empy import bragg_empy
from lumerical_eme import lumerical_eme
from lumerical_fdtd import lumerical_fdtd



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-num_periods',type=int, default="50", help='Number of Periods for Bragg Grating (default: 50)')
    parser.add_argument('-length',type=float, default="0.159", help='Length of each segment of BG, Period = Length * 2 (microns) (default: 0.159)')
    parser.add_argument('-num_wavelengths',type=int, default="10", help='Number of wavelengths to sweep (default: 30)')
    parser.add_argument('-wl_lower',type=float, default="1.5", help='Lower wavelength bound (microns) (default: 1.5)')
    parser.add_argument('-wl_upper',type=float, default="1.6", help='Upper wavelength bound (microns) (default: 1.6)')
    parser.add_argument('-num_modes',type=int, default="1", help='Number of Modes (default: 1)')
    parser.add_argument('-mesh',type=int, default="128", help='Number of mesh points (default: 128)')
    parser.add_argument('-width1',type=float, default="0.46", help='Width of first core block (microns) (default: 0.46)')
    parser.add_argument('-width2',type=float, default="0.54", help='Width of second core block  (microns) (default: 0.54)')
    parser.add_argument('-thickness',type=float, default="0.22", help='Thickness of the core (microns) (default: 0.22)')

    args = parser.parse_args()

    # t_ann = bragg_ann(args)
    t_lumerical_eme = lumerical_eme(args) # Retry tomorrow with proper licensing issues gone and then make sure to fix the normalization of modes and mode overlap formulation for the other solvers
    t_lumerical = bragg_lumerical(args)
    # t_empy = bragg_empy(args)
    # t_lumerical_fdtd = lumerical_fdtd(args)

    plt.figure()
    # plt.plot(np.linspace(1.5,1.6,args.num_wavelengths),t_ann, label="Neural Networks")
    plt.plot(np.linspace(1.5,1.6,args.num_wavelengths),t_lumerical, label="Lumerical MODE FDE")
    # plt.plot(np.linspace(1.5,1.6,args.num_wavelengths),t_empy, label="Electromagnetic Python")
    plt.plot(np.linspace(1.5,1.6,args.num_wavelengths),t_lumerical_eme, label="Lumerical EME")
    # plt.plot(np.linspace(1.5,1.6,args.num_wavelengths),t_lumerical_fdtd, label="Lumerical FDTD")
    plt.xlabel('Lambda (um)')
    plt.ylabel('Trasmission Power')
    plt.legend()
    plt.grid()
    plt.show()
    # plt.savefig("benchmark")
    # plt.savefig("benchmark.eps",format="eps")

if __name__ == "__main__":
    
    main()