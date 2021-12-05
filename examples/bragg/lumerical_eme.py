import lumapi as lm
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse

def lumerical_eme(args):
    # open file
    if os.path.isfile("api.lms"):
        os.remove("api.lms")
    mode = lm.MODE(hide=True)
    mode.save("api.lms")

    t = []
    for wavelength in np.linspace(args.wl_lower, args.wl_upper, args.num_wavelengths):

        mode.deleteall()

        # define parameters
        vlength = 1e-6 * args.length
        vwidth1 = 1e-6 * args.width1
        vwidth2 = 1e-6 * args.width2
        vthickness = 1e-6 * args.thickness
        vnum_modes = args.num_modes
        vmesh = args.mesh
        vwavelength = 1e-6 * wavelength
        vnum_periods = args.num_periods


        # define cladding
        cladding = mode.addrect()
        cladding.name = "cladding"
        cladding.x = 0
        cladding.x_span = 100*vlength
        cladding.y = 0
        cladding.y_span  = 10*max([vwidth1,vwidth2])
        cladding.z = 0
        cladding.z_span  = 10*vthickness
        cladding.material = "SiO2 (Glass) - Palik"

        # define core block 1
        core1 = mode.addrect()
        core1.name = "core1"
        core1.x = -0.5*vlength
        core1.x_span = vlength
        core1.y = 0
        core1.y_span = vwidth1
        core1.z = 0
        core1.z_span = vthickness
        core1.material = "Si (Silicon) - Palik"

        # define core block 2
        core2 = mode.addrect()
        core2.name = "core2"
        core2.x = 0.5*vlength
        core2.x_span = vlength
        core2.y = 0
        core2.y_span = vwidth2
        core2.z = 0
        core2.z_span = vthickness
        core2.material = "Si (Silicon) - Palik"

        # setup eme
        eme = mode.addeme()
        mode.set("wavelength",vwavelength)
        mode.set("mesh cells y",vmesh)
        mode.set("mesh cells z",vmesh)
        mode.set("x min",-vlength)
        mode.set("y",0)
        mode.set("y span",2e-6)
        mode.set("z",0)
        mode.set("z span",2e-6)
        mode.set("allow custom eigensolver settings",1)
        mode.set("cells",2)
        mode.set("group spans",2*vlength)
        mode.set("modes",vnum_modes)
        mode.set("periods",vnum_periods)
        mode.set("y min bc","PML")
        mode.set("y max bc","PML")
        mode.set("z min bc","PML")
        mode.set("z max bc","PML")

        # run
        mode.run()
        mode.emepropagate()

        # postprocess
        internal_s_matrix = mode.getresult("EME","internal s matrix")
        transmission_coefficient = np.abs(internal_s_matrix[0,vnum_modes])**2
        t.append(transmission_coefficient)
        mode.switchtolayout()

    mode.close()
    return t


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('-num_periods',type=int, default="50", help='Number of Periods for Bragg Grating (default: 50)')
    parser.add_argument('-length',type=float, default="0.159", help='Length of each segment of BG, Period = Length * 2 (microns) (default: 0.159)')
    parser.add_argument('-num_wavelengths',type=int, default="30", help='Number of wavelengths to sweep (default: 30)')
    parser.add_argument('-wl_lower',type=float, default="1.5", help='Lower wavelength bound (microns) (default: 1.5)')
    parser.add_argument('-wl_upper',type=float, default="1.6", help='Upper wavelength bound (microns) (default: 1.6)')
    parser.add_argument('-num_modes',type=int, default="1", help='Number of Modes (default: 1)')
    parser.add_argument('-mesh',type=int, default="128", help='Number of mesh points (default: 128)')
    parser.add_argument('-width1',type=float, default="0.46", help='Width of first core block (microns) (default: 0.46)')
    parser.add_argument('-width2',type=float, default="0.54", help='Width of second core block  (microns) (default: 0.54)')
    parser.add_argument('-thickness',type=float, default="0.22", help='Thickness of the core (microns) (default: 0.22)')

    args = parser.parse_args()
    t = lumerical_eme(args)

    #Plot the results
    plt.plot(np.linspace(args.wl_lower,args.wl_upper, args.num_wavelengths), t)
    plt.title("Grating freq sweep nperiods=" + str(args.num_periods))
    plt.xlabel("Wavelength (microns)")
    plt.ylabel("t")
    plt.show()

if __name__ == "__main__":
    
    main()
